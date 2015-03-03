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


def print_mb(slots, classes, vocab_rev, mb, prediction):
    x, x_score, x_switch, x_actor, y_seq_id, y_time, y_labels = mb

    labels = {}
    pred_id = {}
    for i, (seq_id, time), lbls in enumerate(zip(zip(y_seq_id, y_time),
                                              *y_labels)):
        labels[(seq_id, time)] = lbls
        pred_id[(seq_id, time)] = i

    example = []
    for dialog_id, dialog in enumerate(zip(*x)):
        print
        for t, w in enumerate(dialog):
            print vocab_rev[w],

            curr_ndx = (dialog_id, t)
            if curr_ndx in labels:
                curr_label = labels[curr_ndx]
                curr_pred = [prediction[i][pred_id[curr_ndx]]
                             for i, _ in enumerate(slots)]

                print
                print curr_label
                print curr_pred




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
        data = model.prepare_data_train(mb, slots)
        minibatches.append(data)

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


def visualize_mb(model, mb):
    prediction = model._predict(*mb)

    #print_mb(xtd_v, prediction_valid)


def eval_model(model, slots, classes, xtd_t, xtd_v, train_data, valid_data,
               class_groups,
               best_acc, best_acc_train, tracker_valid, tracker_train,
               track_log):
    prediction_valid = model._predict(*valid_data)
    visualize_prediction(xtd_v, prediction_valid)

    eval_train = train_data is not None

    if eval_train:
        prediction_train = model._predict(*train_data)
        pass
    else:
        prediction_train = None

    logging.info('Results:')
    for group_name, slot_selection in class_groups.iteritems():
        joint_slot_name = 'joint_%s' % str(group_name)
        if eval_train:
            train_conf_mats = compute_stats(slots, slot_selection, classes,
                                        prediction_train,
                                        train_data['y_labels'],
                                        joint_slot_name)

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
            if eval_train:
                p.tab(30)
                acc = int(train_conf_mats[slot].accuracy() * 100)
                best_acc_train[slot] = max(best_acc_train[slot], acc)
                p.print_out("%d (%d)" % (acc, best_acc_train[slot]))
            logging.info(p.render())

            cmat = valid_conf_mats[slot].mat
            if eval_train:
                cmat_train = train_conf_mats[slot].mat
            else:
                cmat_train = None


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
                if eval_train:
                    p, r, total_i = compute_prt(cmat_train, i)
                    pp.tab(43)
                    pp.print_out("%10s P(%3d) R(%3d) Total(%4d)" % (cls_name[:10],
                                                         int(p),
                                                         int(r),
                                                         total_i))
                logging.info(pp.render())


    _, accuracy = tracker_valid.track(tracking_log_file_name=track_log)
    logging.info('Tracking accuracy: %d (valid)' % int(accuracy * 100))

    #if eval_train:
    #_, accuracy_train = tracker_train.track(
    #    tracking_log_file_name=track_log + ".train")
    #logging.info('Tracking accuracy: %d (train)' % int(accuracy_train *
    # 100))

    return accuracy

class TrainingStats(object):
    def __init__(self):
        self.data = defaultdict(list)

    def insert(self, **kwargs):
        for key, val in kwargs.iteritems():
            if type(val) is np.ndarray:
                val = float(val)
            self.data[key].append(val)

    def mean(self, arg):
        return np.array(self.data[arg]).mean()


def _get_example_list(minibatches, sorted_items, xtd_t):
    examples = []
    for ii, (i, loss) in enumerate(sorted_items):
        _, d = minibatches[i]

        x, x_score, x_switch, x_actor, y_seq_id, y_time, y_labels = d

        example = []
        for d in zip(*x):
            ln = ""
            for w in d:
                ln += xtd_t.vocab_rev[w]
                ln += " "
            example.append(ln)
        examples.append(example)
    return examples


def get_extreme_examples(mb_loss, minibatches, xtd_t):
    sorted_items = sorted(mb_loss.items(), key=lambda e: -e[1])
    worst_mb_ndxs = sorted_items[:5]
    worst_examples = _get_example_list(minibatches, worst_mb_ndxs, xtd_t)
    best_mb_ndxs = sorted_items[-5:]
    best_examples = _get_example_list(minibatches, best_mb_ndxs, xtd_t)

    return (worst_examples, worst_mb_ndxs), (best_examples, best_mb_ndxs)


def main(args_lst,
         eid, experiment_path, out, valid_after,
         load_params, save_params,
         debug, track_log,
         n_cells, emb_size, x_include_score,
         n_epochs, lr, opt_type, momentum,
         mb_size, mb_mult_data,
         oclf_n_hidden, oclf_n_layers, oclf_activation,
         rnn_n_layers,
         lstm_peepholes, lstm_bidi,
         p_drop, init_emb_from, input_n_layers, input_n_hidden,
         input_activation,
         eval_on_full_train, enable_input_ftrs, enable_branch_exp):

    output_dir = init_env(out)
    mon_train = TrainingStats()
    mon_valid = TrainingStats()
    mon_extreme_examples = TrainingStats()
    stats_obj = dict(
        train=mon_train.data,
        mon_extreme_examples=mon_extreme_examples.data,
        args=args_lst
    )

    logging.info('XTrack has been started.')
    logging.info('Output dir: %s' % output_dir)
    logging.info('Initializing random seed to 0.')
    random.seed(0)
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

    logging.info('Building model.')
    model = Model(slots=slots,
                  slot_classes=xtd_t.classes,
                  emb_size=emb_size,
                  x_include_score=x_include_score,
                  enable_input_ftrs=enable_input_ftrs,
                  n_cells=n_cells,
                  n_input_tokens=n_input_tokens,
                  oclf_n_hidden=oclf_n_hidden,
                  oclf_n_layers=oclf_n_layers,
                  oclf_activation=oclf_activation,
                  debug=debug,
                  rnn_n_layers=rnn_n_layers,
                  lstm_peepholes=lstm_peepholes,
                  lstm_bidi=lstm_bidi,
                  opt_type=opt_type,
                  momentum=momentum,
                  p_drop=p_drop,
                  init_emb_from=init_emb_from,
                  vocab=xtd_t.vocab,
                  input_n_layers=input_n_layers,
                  input_n_hidden=input_n_hidden,
                  input_activation=input_activation,
                  token_features=xtd_t.token_features,
                  enable_branch_exp=enable_branch_exp
    )

    logging.info('Rebuilding took: %.1f' % (time.time() - t))

    if load_params:
        logging.info('Loading parameters from: %s' % load_params)
        model.load_params(load_params)

    tracker_valid = XTrack2DSTCTracker(xtd_v, model)
    tracker_train = XTrack2DSTCTracker(xtd_t, model)

    valid_data_y = model.prepare_data_train(xtd_v.sequences, slots)
    valid_data = model.prepare_data_predict(xtd_v.sequences, slots)
    if not eval_on_full_train:
        selected_train_seqs = []
        for i in range(100):
            ndx = random.randint(0, len(xtd_t.sequences) - 1)
            selected_train_seqs.append(xtd_t.sequences[ndx])
    else:
        selected_train_seqs = xtd_t.sequences

    train_data = model.prepare_data_train(selected_train_seqs, slots)
    joint_slots = ['joint_%s' % str(grp) for grp in class_groups.keys()]
    best_acc = {slot: 0 for slot in xtd_v.slots + joint_slots}
    best_acc_train = {slot: 0 for slot in xtd_v.slots + joint_slots}
    best_tracking_acc = 0.0
    n_valid_not_increased = 0
    et = None
    seqs = list(xtd_t.sequences)
    seqs = seqs * mb_mult_data
    random.shuffle(seqs)
    minibatches = prepare_minibatches(seqs, mb_size, model, slots)
    minibatches = zip(itertools.count(), minibatches)
    logging.info('We have %d minibatches.' % len(minibatches))

    example_cntr = 0
    timestep_cntr = 0
    stats = TrainingStats()
    mb_histogram = defaultdict(int)
    mb_ids = range(len(minibatches))
    mb_to_go = []
    mb_bad = []

    epoch = 0

    init_valid_loss = model._loss(*valid_data_y)
    logging.info('Initial valid loss: %.10f' % init_valid_loss)

    if not valid_after:
        valid_after = len(seqs)

    mb_loss = {}
    last_valid = 0
    last_inline_print = time.time()
    last_inline_print_cnt = 0
    best_track_score = 0.0
    while True:
        if len(mb_to_go) == 0:
            mb_to_go = list(mb_ids)
            epoch += 1

            if n_epochs > 0 and n_epochs < epoch:
                break

        mb_ndx = random.choice(mb_to_go)
        mb_to_go.remove(mb_ndx)

        #mb_id, mb_data = random.choice(minibatches)
        mb_id, mb_data = minibatches[mb_ndx]
        mb_histogram[mb_ndx] += 1
        #if et is not None:
        #    epoch_time = time.time() - et
        #else:
        #    epoch_time = -1.0
        #logging.info('Epoch #%d (last epoch took %.1fs) (seen %d examples)' %
        #             (i, epoch_time, example_cntr ))

        #et = time.time()
        mb_done = 0
        t = time.time()
        (loss, update_ratio) = model._train(lr, *mb_data)
        mb_loss[mb_ndx] = loss
        t = time.time() - t
        stats.insert(loss=loss, update_ratio=update_ratio, time=t)

        x = mb_data[0]
        example_cntr += x.shape[1]
        timestep_cntr += x.shape[0]
        mb_done += 1

        if time.time() - last_inline_print > 1.0:
            last_inline_print = time.time()
            inline_print("     %6d examples, %4d examples/s" % (
                example_cntr,
                example_cntr - last_inline_print_cnt
            ))
            last_inline_print_cnt = example_cntr

        if (example_cntr - last_valid) >= valid_after:
            inline_print("")
            last_valid = example_cntr
            params_file = os.path.join(output_dir, 'params.%.10d.p' %
                                       example_cntr)
            logging.info('Saving parameters: %s' % params_file)
            model.save_params(params_file)
            tracking_acc = 0.0

            valid_loss = model._loss(*valid_data_y)
            update_ratio = stats.mean('update_ratio')
            update_ratio = stats.mean('update_ratio')

            _, track_score = tracker_valid.track(track_log)

            best_track_score = max(track_score, best_track_score)

            logging.info('Valid loss:         %10.2f' % valid_loss)
            logging.info('Valid tracking acc: %10.2f %%' % (track_score * 100))
            logging.info('Best tracking acc:  %10.2f %%' % (best_track_score * 100))
            logging.info('Train loss:         %10.2f' % stats.mean('loss'))
            logging.info('Mean update ratio:  %10.6f' % update_ratio)
            logging.info('Mean mb time:       %10.4f' % stats.mean('time'))
            logging.info('Epoch:              %10d (%d mb remain)' % (epoch,
                                                                     len(mb_to_go)))
            logging.info('Example:            %10d' % example_cntr)


            mon_train.insert(
                time=time.time(),
                example=example_cntr,
                timestep_cntr=timestep_cntr,
                mb_id=mb_id,
                train_loss=stats.mean('loss'),
                valid_loss=valid_loss,
                update_ratio=stats.mean('update_ratio'),
                tracking_acc=tracking_acc
            )

            stats_path = os.path.join(output_dir, 'stats.json')
            with open(stats_path, 'w') as f_out:
                json.dump(stats_obj, f_out)
                os.system('ln -f -s "%s" "xtrack2_vis/stats.json"' %
                          os.path.join('..', stats_path))

            stats = TrainingStats()

    params_file = os.path.join(output_dir, 'params.final.p')
    logging.info('Saving final params to: %s' % params_file)
    model.save_params(params_file)

    return best_tracking_acc


def build_argument_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path')

    parser.add_argument('--eid', default='xtrack_experiment')
    parser.add_argument('--valid_after', default=None, type=int)

    # Experiment params.
    parser.add_argument('--load_params', default=None)
    parser.add_argument('--save_params', default=None)
    parser.add_argument('--out', default='xtrack2_out')

    # XTrack params.
    parser.add_argument('--n_epochs', default=0, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--p_drop', default=0.0, type=float)
    parser.add_argument('--opt_type', default='rprop', type=str)
    parser.add_argument('--mb_size', default=16, type=int)
    parser.add_argument('--mb_mult_data', default=1, type=int)

    parser.add_argument('--n_cells', default=5, type=int)
    parser.add_argument('--emb_size', default=7, type=int)
    parser.add_argument('--x_include_score', default=False, action='store_true')
    parser.add_argument('--enable_input_ftrs', default=False,
                        action='store_true')
    parser.add_argument('--init_emb_from', default=None, type=str)

    parser.add_argument('--input_n_hidden', default=32, type=int)
    parser.add_argument('--input_n_layers', default=0, type=int)
    parser.add_argument('--input_activation', default="sigmoid", type=str)

    parser.add_argument('--oclf_n_hidden', default=32, type=int)
    parser.add_argument('--oclf_n_layers', default=0, type=int)
    parser.add_argument('--oclf_activation', default="tanh", type=str)

    parser.add_argument('--rnn_n_layers', default=1, type=int)

    parser.add_argument('--lstm_peepholes', default=False,
                        action='store_true')
    parser.add_argument('--lstm_bidi', default=False,
                        action='store_true')


    parser.add_argument('--debug', default=False,
                        action='store_true')
    parser.add_argument('--track_log', default='track_log.txt', type=str)
    parser.add_argument('--eval_on_full_train', default=False,
                        action='store_true')

    parser.add_argument('--enable_branch_exp', default=False,
                        action='store_true')

    return parser

if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    pdb_on_error()
    args_lst = list(sorted(vars(args).iteritems()))
    main(args_lst, **vars(args))
