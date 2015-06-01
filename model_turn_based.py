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
        x_scores = T.matrix()
        y_seq_id = tt.ivector()
        y_time = tt.ivector()
        y_label = {}
        for slot in slots:
            y_label[slot] = tt.ivector(name='y_label_%s' % slot)

        input_args = [x, x_scores]
        turns = TurnSquasher(len(self.vocab))
        turns.connect(IdentityInput(x, 1), x_scores * 0 + 1, y_time, y_seq_id)

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
        clipnorm = 5.0
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

    def visualize(self, seqs, slots):
        model_data = self._prepare_data(seqs, slots, with_labels=False)
        preds = self._predict(*model_data)
        pred_ptr = 0

        vocab_rev = {val: key for key, val in self.vocab.iteritems()}
        slot_classes_rev = {}
        for slot in slots:
            slot_classes_rev[slot] = {ndx: val for val, ndx in self.slot_classes[slot].iteritems()}

        with open('/tmp/vis.html', 'w') as f_out:
            for i, dialog in enumerate(seqs):
                d_id = dialog['id']
                true_data = dialog['true_input']
                data = dialog['data']
                lbls = {}
                for lbl in dialog['labels']:
                    lbls[lbl['time']] = lbl

                print >>f_out, "<h1>Dialog %d: %s</h1>" % (i, d_id)
                print >>f_out, "<p>"
                print >>f_out, "<ul>"
                for s in true_data:
                    print >>f_out, "<li>%s</li>" % s
                print >>f_out, "</ul>"
                print >>f_out, "</p>"

                print >>f_out, "<h2>Data</h2>"
                print >>f_out, "<ul>"
                for t, w in enumerate(data):
                    print >>f_out, "<li>%s</li>" % vocab_rev[w]

                    if t in lbls:
                        for slot_ndx, slot in enumerate(slots):
                            curr_pred = preds[slot_ndx][pred_ptr].argmax()
                            curr_pred_p = preds[slot_ndx][pred_ptr].max()
                            print >>f_out, "<li>PRD: %s %.2f</li>" % (slot_classes_rev[slot][curr_pred], curr_pred_p, )

                            true_val = lbls[t]['slots'][slot]
                            print >>f_out, "<li>LBL: %s</li>" % slot_classes_rev[slot][true_val]

                            if curr_pred == true_val or (true_val == 0 and curr_pred_p < 0.5):
                                print >>f_out, "<li>PRED_GOOD</li>"
                            else:
                                print >>f_out, "<li>PRED_BAD</li>"

                        pred_ptr += 1
                print >>f_out, "</ul>"







    def _prepare_data(self, seqs, slots, with_labels=True, debug_data=False):
        x = []
        x_scores = []
        y_seq_id = []
        y_time = []
        y_labels = [[] for slot in slots]
        y_weights = []
        for item in seqs:
            data = item['data']
            score = item['data_score']

            if type(data[0]) is list:
                assert len(data[0]) == 1
                data = [i[0] for i in data]

            x.append(data)
            x_scores.append(score)

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
        x_scores = padded(x_scores).transpose(1, 0)
        x_scores = np.exp(x_scores)

        data = [x, x_scores]
        data.extend([y_seq_id, y_time])
        if with_labels:
            data.extend(y_labels)

        return tuple(data)


