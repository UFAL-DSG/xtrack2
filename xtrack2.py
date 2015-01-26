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
theano.config.allow_gc=False
theano.scan.allow_gc=False
#theano.config.profile=True
#theano.config.mode = 'FAST_COMPILE'
theano.config.mode = 'FAST_RUN'

from passage.iterators import (padded, SortedPadded)
from passage.utils import iter_data
from xtrack_data2 import XTrackData2
from utils import (pdb_on_error, ConfusionMatrix, P)
from model import Model


def compute_stats(slots, classes, prediction, y, prev_conf_mats=None):
    if not prev_conf_mats:
        conf_mats = {}
    else:
        conf_mats = prev_conf_mats

    for slot in slots:
        conf_mats[slot] = ConfusionMatrix(len(classes[slot]))

    for i, (slot, pred) in enumerate(zip(slots, prediction)):
        slot_y = y[i]
        slot_y_hat = np.argmax(pred, axis=1)

        conf_mats[slot].batchAdd(slot_y, slot_y_hat)

    return conf_mats


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


def main(experiment_path, out, n_cells, visualize_every, emb_size,
         n_epochs, lr, opt_type, gradient_clipping, model_file,
         final_model_file, mb_size,
         eid, n_neg_samples, rebuild_model, desc, rinit_scale,
         rinit_scale_emb, init_scale_gates_bias, oclf_n_hidden):
    out = init_env(out)

    logging.info('XTrack has been started.')
    logging.info(str(sys.argv))

    train_path = os.path.join(experiment_path, 'train.json')
    valid_path = os.path.join(experiment_path, 'valid.json')
    # test_path = os.path.join(experiment_path, 'test.hdf5')

    xtd_t = XTrackData2.load(train_path)
    xtd_v = XTrackData2.load(valid_path)

    #import ipdb; ipdb.set_trace()

    slots = xtd_t.slots
    classes = xtd_t.classes
    n_input_tokens = len(xtd_t.vocab)

    t = time.time()
    if rebuild_model:
        logging.info('Rebuilding model.')
        model = Model(slots, xtd_t.classes, 10, n_input_tokens, None)
        model.save(model_file)
        logging.info('Rebuilding took: %.1f' % (time.time() - t))
    else:
        logging.info('Loading model from: %s' % model_file)
        model = Model.load(model_file)
        logging.info('Loading took: %.1f' % (time.time() - t))

    seqs = xtd_t.sequences
    random.shuffle(seqs)

    seqs_mb = iter_data(seqs, size=mb_size)
    minibatches = []
    for mb in seqs_mb:
        x = []
        y_seq_id = []
        y_time = []
        y_labels = [[] for slot in slots]
        for item in mb:
            x.append(item['data'])
            for label in item['labels']:
                y_seq_id.append(len(x) - 1)
                y_time.append(label['time'])

                for i, slot in enumerate(slots):
                    y_labels[i].append(label['slots'][slot])

        x = padded(x).transpose(1, 0)


        minibatches.append((x, y_seq_id, y_time, y_labels, ))

    prev_conf_mats = None
    for i in range(n_epochs):
        logging.info('Iteration #%d' % i)
        random.shuffle(minibatches)
        avg_loss = 0.0
        for mb_id, (x, y_seq_id, y_time, y_labels) in enumerate(minibatches):
            t = time.time()
            loss = model._train(x, y_seq_id, y_time, *y_labels)
            t = time.time() - t
            avg_loss += loss
            vlog(' > ',
                 ('minibatch', mb_id, ),
                 ('loss', "%.2f" % loss),
                 ('time', "%.1f" % t),
                 ('xsize', str(x.shape)),
                 ('ysize', len(y_seq_id)),
                 separator=" ",)

            prediction = model._predict(x, y_seq_id, y_time)

            prev_conf_mats = compute_stats(slots, classes, prediction,
                                           y_labels,
                                           prev_conf_mats=prev_conf_mats)
        avg_loss = avg_loss / len(minibatches)
        logging.info('Mean loss: %.2f' % avg_loss)

        logging.info('Results:')
        for slot in slots:
            p = P()
            p.tab(3)
            p.print_out(slot)
            p.tab(15)
            p.print_out("%d" % int(prev_conf_mats[slot].accuracy() * 100))
            logging.info(p.render())


    logging.info('Saving final model to: %s' % final_model_file)
    model.save(final_model_file)

if __name__ == '__main__':


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
    parser.add_argument('--desc', default='', required=False)
    parser.add_argument('--visualize_every', default=60, type=int)

    # XTrack params.
    parser.add_argument('--n_cells', default=5, type=int)
    parser.add_argument('--emb_size', default=7, type=int)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--lr', default=1.0, type=float)
    parser.add_argument('--opt_type', default='rprop', type=str)
    parser.add_argument('--gradient_clipping', default=None, type=float)
    #parser.add_argument('--unit_act', )
    parser.add_argument('--n_neg_samples', default=2, type=int)
    parser.add_argument('--rinit_scale', default=0.1, type=float)
    parser.add_argument('--rinit_scale_emb', default=1.0, type=float)
    parser.add_argument('--init_scale_gates_bias', default=0.0, type=float)
    parser.add_argument('--mb_size', default=32, type=int)

    parser.add_argument('--oclf_n_hidden', default=30, type=int)

    args = parser.parse_args()
    #climate.call(main)
    pdb_on_error()
    main(**vars(args))