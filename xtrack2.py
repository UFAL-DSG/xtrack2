import itertools
import os
import sys
import logging
import numpy as np
import random
import theano
import theano.gradient
import time

theano.config.floatX = 'float32'
#theano.config.allow_gc=False
#theano.scan.allow_gc=False
#theano.config.profile=True
#theano.config.mode = 'FAST_COMPILE'
theano.config.mode = 'FAST_RUN'

from passage.iterators import (padded, SortedPadded)
from passage.utils import iter_data
from xtrack_data2 import XTrackData2
from utils import (pdb_on_error, ConfusionMatrix, P)
from model import Model
from xtrack2_dstc_tracker import XTrack2DSTCTracker


def compute_stats(slots, slot_selection, classes, prediction, y,
                  joint_slot_name):
    conf_mats = {}
    conf_mats[joint_slot_name] = ConfusionMatrix(2)
    for slot in slots:
        if slot in slot_selection:
            conf_mats[slot] = ConfusionMatrix(len(classes[slot]))


    joint_correct = np.array([True for _ in prediction[0]])
    joint_all = np.array([True for _ in prediction[0]])
    for i, (slot, pred) in enumerate(zip(slots, prediction)):
        if slot in slot_selection:
            slot_y = y[i]
            slot_y_hat = np.argmax(pred, axis=1)

            conf_mats[slot].batchAdd(slot_y, slot_y_hat)

            joint_correct &= (slot_y == slot_y_hat)

    conf_mats[joint_slot_name].batchAdd(joint_all, joint_correct)
    return conf_mats


def visualize_prediction(xtd, prediction):
    #x = data['x'].transpose(1, 0)
    pred_ptr = 0

    classes_rev = {}
    for slot in xtd.slots:
        classes_rev[slot] = {val: key
                             for key, val in xtd.classes[slot].iteritems()}

    for d_id, dialog in enumerate(xtd.sequences[:3]):
        print ">> Dialog %d" % d_id, "=" * 30

        labeling = {}
        for label in dialog['labels']:
            pred_label = {}
            for i, slot in enumerate(xtd.slots):
                pred = prediction[i][pred_ptr]
                pred_label[slot] = np.argmax(pred)
            pred_ptr += 1

            labeling[label['time']] = (label['slots'], pred_label)

        print " T:",
        for i, word_id in enumerate(dialog['data']):
            print xtd.vocab_rev[word_id],
            if i in labeling:
                print
                for slot in xtd.slots:
                    lbl, pred_lbl = labeling[i]
                    p = P()
                    p.print_out("    * ")
                    p.print_out(slot)
                    p.tab(20)
                    p.print_out(classes_rev[slot][lbl[slot]])
                    p.tab(40)
                    p.print_out(classes_rev[slot][pred_lbl[slot]])
                    print p.render()
                print " U:",
        print



def vlog(txt, *args, **kwargs):
    separator = kwargs.pop('separator', '\n')
    res = [txt]
    for k, v in args:
        res.append('\t%s(%s)' % (k, str(v)))
    for k, v in sorted(kwargs.iteritems()):
        res.append('\t%s(%s)' % (k, str(v)))
    logging.info(separator.join(res))


def init_output_dir(out_dir):
    cntr = 1
    orig_out_dir = out_dir
    while os.path.exists(out_dir):
        out_dir = orig_out_dir + '_' + str(cntr)
        cntr += 1

    os.makedirs(out_dir)
    return out_dir


def init_env(output_dir):
    output_dir = init_output_dir(output_dir)

    theano.config.compute_test_value = 'off'
    theano.config.allow_gc=False
    theano.scan.allow_gc=False

    # Setup logging.
    logger = logging.getLogger('XTrack')
    logger.setLevel(logging.DEBUG)

    logging_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    formatter = logging.Formatter(logging_format)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.root = logger  #

    return output_dir


def prepare_minibatches(seqs, mb_size, model, slots):
    minibatches = []
    seqs_mb = iter_data(seqs, size=mb_size)
    for mb in seqs_mb:
        res = model.prepare_data(mb, slots)

        x = res['x']
        y_seq_id = res['y_seq_id']
        y_time = res['y_time']
        y_labels = res['y_labels']

        minibatches.append((x, y_seq_id, y_time, y_labels, ))

    return minibatches


def eval_model(model, slots, classes, xtd_t, xtd_v, train_data, valid_data,
               class_groups,
               best_acc, best_acc_train, tracker_valid, tracker_train,
               track_log):
    prediction_valid = model._predict(
        valid_data['x'],
        valid_data['y_seq_id'],
        valid_data['y_time']
    )
    prediction_train = model._predict(
        train_data['x'],
        train_data['y_seq_id'],
        train_data['y_time']
    )

    visualize_prediction(xtd_v, prediction_valid)
    visualize_prediction(xtd_t, prediction_train)

    logging.info('Results:')
    for group_name, slot_selection in class_groups.iteritems():
        joint_slot_name = 'joint_%s' % str(group_name)
        train_conf_mats = compute_stats(slots, slot_selection, classes,
                                        prediction_train,
                                        train_data['y_labels'], joint_slot_name)

        valid_conf_mats = compute_stats(slots, slot_selection, classes,
                                        prediction_valid,
                                        valid_data['y_labels'], joint_slot_name)

        for slot in slot_selection + [joint_slot_name]:
            p = P()
            p.tab(3)
            p.print_out(slot)
            p.tab(15)
            acc = int(valid_conf_mats[slot].accuracy() * 100)
            best_acc[slot] = max(best_acc[slot], acc)
            p.print_out("%d (%d)" % (acc, best_acc[slot]))
            p.tab(25)
            acc = int(train_conf_mats[slot].accuracy() * 100)
            best_acc_train[slot] = max(best_acc_train[slot], acc)
            p.print_out("%d (%d)" % (acc, best_acc_train[slot]))
            logging.info(p.render())

    _, accuracy = tracker_valid.track(tracking_log_file_name=track_log)
    _, accuracy_train = tracker_train.track(
        tracking_log_file_name=track_log + ".train")
    logging.info('Tracking accuracy: %d (valid)' % int(accuracy * 100))
    logging.info('Tracking accuracy: %d (train)' % int(accuracy_train * 100))


def main(experiment_path, out, n_cells, emb_size,
         n_epochs, lr, opt_type, model_file,
         final_model_file, mb_size,
         eid, rebuild_model, oclf_n_hidden,
         oclf_n_layers, oclf_activation, debug, track_log, lstm_n_layers,
         p_drop, init_emb_from):
    out = init_env(out)

    logging.info('XTrack has been started.')
    logging.info('ARGV: %s' % str(sys.argv))
    logging.info('Experiment path: %s' % experiment_path)

    train_path = os.path.join(experiment_path, 'train.json')
    xtd_t = XTrackData2.load(train_path)

    valid_path = os.path.join(experiment_path, 'valid.json')
    xtd_v = XTrackData2.load(valid_path)

    slots = xtd_t.slots
    classes = xtd_t.classes
    class_groups = xtd_t.slot_groups
    n_input_tokens = len(xtd_t.vocab)

    t = time.time()
    if rebuild_model:
        logging.info('Rebuilding model.')
        model = Model(slots=slots,
                      slot_classes=xtd_t.classes,
                      emb_size=emb_size,
                      n_cells=n_cells,
                      n_input_tokens=n_input_tokens,
                      oclf_n_hidden=oclf_n_hidden,
                      oclf_n_layers=oclf_n_layers,
                      oclf_activation=oclf_activation,
                      lr=lr,
                      debug=debug,
                      lstm_n_layers=lstm_n_layers,
                      opt_type=opt_type,
                      p_drop=p_drop,
                      init_emb_from=init_emb_from,
                      vocab=xtd_t.vocab
        )

        model.save(model_file)
        logging.info('Rebuilding took: %.1f' % (time.time() - t))
    else:
        logging.info('Loading model from: %s' % model_file)
        model = Model.load(model_file)
        logging.info('Loading took: %.1f' % (time.time() - t))

    tracker = XTrack2DSTCTracker(xtd_v, model)
    tracker_train = XTrack2DSTCTracker(xtd_t, model)

    minibatches = prepare_minibatches(xtd_t.sequences, mb_size, model, slots)
    minibatches = zip(itertools.count(), minibatches)
    logging.info('We have %d minibatches.' % len(minibatches))

    valid_data = model.prepare_data(xtd_v.sequences, slots)
    train_data = model.prepare_data(xtd_t.sequences, slots)

    joint_slots = ['joint_%s' % str(grp) for grp in class_groups.keys()]
    best_acc = {slot: 0 for slot in xtd_v.slots + joint_slots}
    best_acc_train = {slot: 0 for slot in xtd_v.slots + joint_slots}
    for i in range(n_epochs):
        logging.info('Epoch #%d' % i)
        random.shuffle(minibatches)
        avg_loss = 0.0

        for mb_id, (x, y_seq_id, y_time, y_labels) in minibatches:
            t = time.time()
            (loss, update_ratio) = model._train(x, y_seq_id, y_time, *y_labels)
            t = time.time() - t

            avg_loss += loss

            vlog(' > ',
                 ('minibatch', mb_id, ),
                 ('loss', "%.2f" % loss),
                 ('ratio', "%.5f" % update_ratio),
                 ('time', "%.1f" % t),
                 ('xsize', str(x.shape)),
                 ('ysize', len(y_seq_id)),
                 separator=" ",)
        avg_loss = avg_loss / len(minibatches)
        logging.info('Mean loss: %.2f' % avg_loss)

        eval_model(model=model, train_data=train_data, valid_data=valid_data,
                   class_groups=class_groups, slots=slots, classes=classes,
                   xtd_t=xtd_t, xtd_v=xtd_v,
                   best_acc=best_acc, best_acc_train=best_acc_train,
                   tracker_train=tracker_train, tracker_valid=tracker,
                   track_log=track_log)


    logging.info('Saving final model to: %s' % final_model_file)
    model.save(final_model_file)

if __name__ == '__main__':
    random.seed(0)


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path')

    parser.add_argument('--eid', default='xtrack_experiment')

    # Experiment params.
    parser.add_argument('--rebuild_model', action='store_true', default=False,
                        required=False)
    parser.add_argument('--model_file', default='xtrack_model.pickle')
    parser.add_argument('--final_model_file',
                        default='xtrack_model_final.pickle')
    parser.add_argument('--out', required=True)

    # XTrack params.
    parser.add_argument('--n_cells', default=5, type=int)
    parser.add_argument('--emb_size', default=7, type=int)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--p_drop', default=0.0, type=float)
    parser.add_argument('--opt_type', default='rprop', type=str)
    parser.add_argument('--mb_size', default=16, type=int)
    parser.add_argument('--init_emb_from', default=None, type=str)

    parser.add_argument('--oclf_n_hidden', default=32, type=int)
    parser.add_argument('--oclf_n_layers', default=2, type=int)
    parser.add_argument('--oclf_activation', default="tanh", type=str)
    parser.add_argument('--lstm_n_layers', default=1, type=int)

    parser.add_argument('--debug', default=False,
                        action='store_true')
    parser.add_argument('--track_log', default='rprop', type=str)


    args = parser.parse_args()
    #climate.call(main)
    pdb_on_error()
    main(**vars(args))
