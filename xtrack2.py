from collections import defaultdict
import itertools
import json
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
#theano.config.linker = 'py'
theano.config.mode = 'FAST_RUN'

from passage.iterators import (padded, SortedPadded)
from passage.utils import iter_data
from xtrack_data2 import XTrackData2
from utils import (pdb_on_error, ConfusionMatrix, P, inline_print)
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
        last_score = None
        for i, word_id, score in zip(itertools.count(),
                                     dialog['data'],
                                     dialog['data_score']):
            if score != last_score:
                print "%4.2f" % score,
                last_score = score

            print xtd.vocab_rev[word_id], #"(%.2f)" % score,
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
        x_score = res['x_score']
        x_switch = res['x_switch']
        x_actor = res['x_actor']
        y_seq_id = res['y_seq_id']
        y_time = res['y_time']
        y_labels = res['y_labels']

        minibatches.append((x, x_score, x_switch, x_actor,
                            y_seq_id, y_time, y_labels, ))

    return minibatches

def compute_prt(cmat, i):
    r = cmat[i,i] * 100.0
    total_i = cmat[i, :].sum()
    if total_i > 0:
        r /= total_i
    else:
        r = 100.0
    p = cmat[i,i] * 100.0
    total_ii = cmat[:, i].sum()
    if total_ii > 0:
        p /= total_ii
    else:
        p = 100.0

    return p, r, total_i


def eval_model(model, slots, classes, xtd_t, xtd_v, train_data, valid_data,
               class_groups,
               best_acc, best_acc_train, tracker_valid, tracker_train,
               track_log):
    prediction_valid = model._predict(
        valid_data['x'],
        valid_data['x_score'],
        valid_data['x_switch'],
        #valid_data['x_actor'],
        valid_data['y_seq_id'],
        valid_data['y_time']
    )
    #prediction_train = model._predict(
    #    train_data['x'],
    #    train_data['x_score'],
    #    train_data['x_switch'],
    #    #train_data['x_actor'],
    #    train_data['y_seq_id'],
    #    train_data['y_time']
    #)

    visualize_prediction(xtd_v, prediction_valid)
    #visualize_prediction(xtd_t, prediction_train)

    logging.info('Results:')
    for group_name, slot_selection in class_groups.iteritems():
        joint_slot_name = 'joint_%s' % str(group_name)
        #train_conf_mats = compute_stats(slots, slot_selection, classes,
        #                                prediction_train,
        #                                train_data['y_labels'],
        # joint_slot_name)

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
            p.print_out("%d (best %d)" % (acc, best_acc[slot]))
            #p.tab(30)
            #acc = int(train_conf_mats[slot].accuracy() * 100)
            #best_acc_train[slot] = max(best_acc_train[slot], acc)
            #p.print_out("%d (%d)" % (acc, best_acc_train[slot]))
            logging.info(p.render())

            cmat = valid_conf_mats[slot].mat
            #cmat_train = train_conf_mats[slot].mat
            if slot != joint_slot_name:
                slot_classes = classes[slot]
            else:
                slot_classes = {'correct': 1, 'incorrect': 0}
            #slot_classes_rev = {v: k for k, v in slot_classes.iteritems()}

            for cls_name, i in sorted(slot_classes.iteritems()):
                p, r, total_i = compute_prt(cmat, i)
                pp = P()
                pp.tab(4)
                pp.print_out("%10s P(%3d) R(%3d) Total(%4d)" % (cls_name[:10],
                                                     int(p),
                                                     int(r),
                                                     total_i))
                #p, r, total_i = compute_prt(cmat_train, i)
                #pp.tab(43)
                #pp.print_out("%10s P(%3d) R(%3d) Total(%4d)" % (cls_name[:10],
                #                                     int(p),
                #                                     int(r),
                #                                     total_i))
                logging.info(pp.render())


    _, accuracy = tracker_valid.track(tracking_log_file_name=track_log)
    #_, accuracy_train = tracker_train.track(
    #    tracking_log_file_name=track_log + ".train")
    logging.info('Tracking accuracy: %d (valid)' % int(accuracy * 100))
    #logging.info('Tracking accuracy: %d (train)' % int(accuracy_train * 100))

    return 0.0 #accuracy

class TrainingStats(object):
    def __init__(self):
        self.data = defaultdict(list)

    def insert(self, **kwargs):
        for key, val in kwargs.iteritems():
            self.data[key].append(val)

    def mean(self, arg):
        return np.array(self.data[arg]).mean()

def main(args_lst, experiment_path, out, n_cells, emb_size,
         n_epochs, lr, opt_type, model_file,
         final_model_file, mb_size,
         eid, rebuild_model, oclf_n_hidden,
         oclf_n_layers, oclf_activation, debug, track_log,
         rnn_type, rnn_n_layers,
         lstm_no_peepholes,
         p_drop, init_emb_from, input_n_layers, input_n_hidden,
         input_activation):
    random.seed(0)

    out = init_env(out)

    logging.info('XTrack has been started.')
    logging.info('Argv: %s' % str(sys.argv))
    logging.info('Effective args:')
    for arg_name, arg_value in args_lst:
        logging.info('    %s: %s' % (arg_name, arg_value))
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
                      debug=debug,
                      rnn_type=rnn_type,
                      rnn_n_layers=rnn_n_layers,
                      lstm_no_peepholes=lstm_no_peepholes,
                      opt_type=opt_type,
                      p_drop=p_drop,
                      init_emb_from=init_emb_from,
                      vocab=xtd_t.vocab,
                      input_n_layers=input_n_layers,
                      input_n_hidden=input_n_hidden,
                      input_activation=input_activation,
                      token_features=xtd_t.token_features
        )

        try:
            model.save(model_file)
        except Exception, e:
            logging.error('Could not save model: %s' % str(e))

        logging.info('Rebuilding took: %.1f' % (time.time() - t))
    else:
        logging.info('Loading model from: %s' % model_file)
        model = Model.load(model_file)
        logging.info('Loading took: %.1f' % (time.time() - t))

    tracker = XTrack2DSTCTracker(xtd_v, model)
    tracker_train = XTrack2DSTCTracker(xtd_t, model)

    valid_data = model.prepare_data(xtd_v.sequences, slots)
    train_data = model.prepare_data(xtd_t.sequences, slots)

    joint_slots = ['joint_%s' % str(grp) for grp in class_groups.keys()]
    best_acc = {slot: 0 for slot in xtd_v.slots + joint_slots}
    best_acc_train = {slot: 0 for slot in xtd_v.slots + joint_slots}
    best_tracking_acc = 0.0
    n_valid_not_increased = 0
    et = None
    seqs = list(xtd_t.sequences)
    random.shuffle(seqs)
    minibatches = prepare_minibatches(seqs, mb_size, model, slots)
    minibatches = zip(itertools.count(), minibatches)
    logging.info('We have %d minibatches.' % len(minibatches))

    example_cntr = 0
    timestep_cntr = 0
    stats = TrainingStats()
    mb_histogram = defaultdict(int)
    while True:
        mb_id, mb_data = random.choice(minibatches)
        mb_histogram[mb_id] += 1
        #if et is not None:
        #    epoch_time = time.time() - et
        #else:
        #    epoch_time = -1.0
        #logging.info('Epoch #%d (last epoch took %.1fs) (seen %d examples)' %
        #             (i, epoch_time, example_cntr ))

        #et = time.time()
        mb_done = 0
        x, x_score, x_switch, x_actor, y_seq_id, y_time, y_labels = mb_data
        t = time.time()
        (loss, update_ratio) = model._train(
            lr, x,
            x_score, x_switch,
            y_seq_id, y_time, *y_labels)
        t = time.time() - t
        example_cntr += x.shape[1]
        timestep_cntr += x.shape[0]
        mb_done += 1

        inline_print("     %6d examples, %10d timesteps" % (
            example_cntr,
            timestep_cntr
        ))

        stats.insert(update_ratio=update_ratio, loss=loss, time=t)
        if example_cntr % 100 == 0:
            logging.info('Processed %d examples, %d timesteps.' % (
                example_cntr, timestep_cntr))
            logging.info('Mean loss:         %10.2f' % stats.mean('loss'))
            logging.info('Mean update ratio: %10.6f' % stats.mean('update_ratio'))
            logging.info('Mean time:         %10.4f' % stats.mean('time'))
            mb_hist_min = min(mb_histogram.values())
            mb_hist_max = max(mb_histogram.values())
            logging.info('MB histogram: min(%d) max(%d)' % (
                mb_hist_min, mb_hist_max))



        if example_cntr % 1000 == 0:
            tracking_acc = eval_model(model=model, train_data=train_data,
                           valid_data=valid_data,
                       class_groups=class_groups, slots=slots, classes=classes,
                       xtd_t=xtd_t, xtd_v=xtd_v,
                       best_acc=best_acc, best_acc_train=best_acc_train,
                       tracker_train=tracker_train, tracker_valid=tracker,
                       track_log=track_log)

            if tracking_acc > best_tracking_acc:
                best_tracking_acc = tracking_acc

            valid_loss = model._loss(
                valid_data['x'],
                valid_data['x_score'],
                valid_data['x_switch'],
                #valid_data['x_actor'],
                valid_data['y_seq_id'],
                valid_data['y_time'],
                *valid_data['y_labels']
            )
            logging.info('Valid loss:         %10.2f' % valid_loss)

            stats = TrainingStats()



    logging.info('Saving final model to: %s' % final_model_file)
    model.save(final_model_file)

    return best_tracking_acc


def build_argument_parser():
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
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--p_drop', default=0.0, type=float)
    parser.add_argument('--opt_type', default='rprop', type=str)
    parser.add_argument('--mb_size', default=16, type=int)

    parser.add_argument('--n_cells', default=5, type=int)
    parser.add_argument('--emb_size', default=7, type=int)
    parser.add_argument('--init_emb_from', default=None, type=str)

    parser.add_argument('--input_n_hidden', default=32, type=int)
    parser.add_argument('--input_n_layers', default=0, type=int)
    parser.add_argument('--input_activation', default="sigmoid", type=str)

    parser.add_argument('--oclf_n_hidden', default=32, type=int)
    parser.add_argument('--oclf_n_layers', default=2, type=int)
    parser.add_argument('--oclf_activation', default="tanh", type=str)

    parser.add_argument('--rnn_type', default='lstm')
    parser.add_argument('--rnn_n_layers', default=1, type=int)

    parser.add_argument('--lstm_no_peepholes', default=False,
                        action='store_true')


    parser.add_argument('--debug', default=False,
                        action='store_true')
    parser.add_argument('--track_log', default='rprop', type=str)

    return parser

if __name__ == '__main__':
    random.seed(0)


    parser = build_argument_parser()


    args = parser.parse_args()
    #climate.call(main)
    pdb_on_error()
    args_lst = list(sorted(vars(args).iteritems()))
    main(args_lst, **vars(args))
