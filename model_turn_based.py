import inspect
import logging
import time

import numpy as np

import theano
import theano.tensor as tt

from passage import updates
from passage.iterators import padded
from passage.layers import *
from passage.model import NeuralModel


class TurnBasedModel(NeuralModel):
    def _log_classes_info(self):
        for slot, vals in self.slot_classes.iteritems():
            logging.info('  %s:' % slot)
            for val, val_ndx in sorted(vals.iteritems(), key=lambda x: x[1]):
                logging.info('    - %s (%d)' % (val, val_ndx))

    def __init__(self, slots, slot_classes, opt_type, mlp_n_hidden,
                 mlp_n_layers, mlp_activation, debug, p_drop, vocab,
                 l1, l2, build_train=True):
        self.store_init_args(locals())

        self.vocab = vocab
        self.slots = slots
        self.slot_classes = slot_classes

        logging.info('We have the following classes:')
        self._log_classes_info()

        x = T.imatrix()
        y_seq_id = tt.ivector()
        y_time = tt.ivector()
        y_label = {}
        for slot in slots:
            y_label[slot] = tt.ivector(name='y_label_%s' % slot)

        input_args = [x]
        turns = TurnSquasher(len(self.vocab))
        turns.connect(IdentityInput(x, 1), y_time, y_seq_id)

        global_mlp = MLP([mlp_n_hidden  ] * mlp_n_layers,
                         [mlp_activation] * mlp_n_layers,
                         [p_drop        ] * mlp_n_layers,
                         name="global_mlp")
        global_mlp.connect(turns)

        costs = []
        predictions = []
        for slot in slots:
            logging.info('Building output classifier for %s.' % slot)
            n_classes = len(slot_classes[slot])
            slot_mlp = Dense(activation='softmax',
                             size=n_classes,
                             p_drop=0.0,
                             name="softmax_%s" % slot)
            slot_mlp.connect(global_mlp)
            predictions.append(slot_mlp.output(dropout_active=False))

            slot_objective = CrossEntropyObjective()
            #slot_objective = SVMObjective()
            slot_objective.connect(
                y_hat_layer=slot_mlp,
                y_true=y_label[slot]
            )
            costs.append(slot_objective)

        cost = SumOut()
        cost.connect(*costs)  #, scale=1.0 / len(slots))
        self.params = params = list(cost.get_params())
        n_params = sum(p.get_value().size for p in params)
        logging.info('This model has %d parameters:' % n_params)
        for param in sorted(params, key=lambda x: x.name):
            logging.info('  - %20s: %10d' % (param.name, param.get_value(

            ).size, ))

        cost_value = cost.output(dropout_active=True)

        assert opt_type == 'sgd'
        lr = tt.scalar('lr')
        clipnorm = 0.0
        reg = updates.Regularizer(l1=l1, l2=l2)
        updater = updates.SGD(lr=lr, clipnorm=clipnorm, regularizer=reg)

        loss_args = list(input_args)
        loss_args += [y_seq_id, y_time]
        loss_args += [y_label[slot] for slot in slots]

        if build_train:
            model_updates = updater.get_updates(params, cost_value)

            train_args = [lr] + loss_args
            update_ratio = updater.get_update_ratio(params, model_updates)

            logging.info('Preparing %s train function.' % opt_type)
            t = time.time()
            self._train = theano.function(train_args, [cost_value, update_ratio],
                                          updates=model_updates)
            logging.info('Preparation done. Took: %.1f' % (time.time() - t))

        self._loss = theano.function(loss_args, cost_value)

        logging.info('Preparing predict function.')
        t = time.time()
        predict_args = list(input_args)
        predict_args += [y_seq_id, y_time]
        self._predict = theano.function(
            predict_args,
            predictions
        )
        logging.info('Done. Took: %.1f' % (time.time() - t))

    def prepare_data_train(self, seqs, slots, debug_data=False):
        return self._prepare_data(seqs, slots, with_labels=True, debug_data=debug_data)

    def prepare_data_predict(self, seqs, slots):
        return self._prepare_data(seqs, slots, with_labels=False)

    def _prepare_y_token_labels_padding(self):
        token_padding = []
        for slot in self.slots:
            token_padding.append(0)
            token_padding.append(0)

        return [token_padding]

    def _prepare_data(self, seqs, slots, with_labels=True, debug_data=False):
        x = []
        y_seq_id = []
        y_time = []
        y_labels = [[] for slot in slots]
        y_weights = []
        for item in seqs:
            data = item['data']

            if type(data[0]) is list:
                assert len(data[0]) == 1
                data = [i[0] for i in data]

            x.append(data)

            labels = item['labels']

            for label in labels:
                y_seq_id.append(len(x) - 1)
                y_time.append(label['time'])

                for i, slot in enumerate(slots):
                    lbl_val = label['slots'][slot]
                    if lbl_val < 0:
                        lbl_val = len(self.slot_classes[slot]) + lbl_val
                    y_labels[i].append(lbl_val)
                y_weights.append(label['score'])

        x = padded(x, is_int=True).transpose(1, 0)


        data = [x]
        data.extend([y_seq_id, y_time])
        if with_labels:
            data.extend(y_labels)

        return tuple(data)


