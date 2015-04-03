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


class SimpleConvModel(NeuralModel):
    def _log_classes_info(self):
        for slot, vals in self.slot_classes.iteritems():
            logging.info('  %s:' % slot)
            for val, val_ndx in sorted(vals.iteritems(), key=lambda x: x[1]):
                logging.info('    - %s (%d)' % (val, val_ndx))

    def __init__(self, slots, slot_classes, emb_size, no_train_emb,
                 x_include_score, x_include_token_ftrs, x_include_mlp,
                 n_input_tokens, n_input_score_bins, n_cells,
                 rnn_n_layers,
                 lstm_peepholes, lstm_bidi, opt_type,
                 oclf_n_hidden, oclf_n_layers, oclf_activation,
                 debug, p_drop,
                 init_emb_from, vocab,
                 input_n_layers, input_n_hidden, input_activation,
                 token_features, token_supervision,
                 momentum, enable_branch_exp, l1, l2, build_train=True):
        args = SimpleConvModel.__init__.func_code.co_varnames[
               :SimpleConvModel.__init__.func_code.co_argcount]
        self.init_args = {}
        for arg in args:
            if arg != 'self':
                self.init_args[arg] = locals()[arg]

        self.vocab = vocab

        self.slots = slots
        self.slot_classes = slot_classes


        logging.info('We have the following classes:')
        self._log_classes_info()

        self.x_include_score = x_include_score
        self.token_supervision = token_supervision

        x = T.imatrix()
        input_args = [x]
        input_token_layer = Embedding(name="emb",
                                      size=emb_size,
                                      n_features=n_input_tokens,
                                      input=x,
                                      static=no_train_emb)

        prev_layer = input_token_layer

        y_seq_id = tt.ivector()
        y_time = tt.ivector()
        y_weight = tt.vector()
        y_label = {}
        for slot in slots:
            y_label[slot] = tt.ivector(name='y_label_%s' % slot)


        if x_include_score:
            x_score = tt.matrix()
            input_args.append(x_score)

        #maxpool = SeqMaxPooling()
        #maxpool.connect(prev_layer, x_score, y_time, y_seq_id)
        #prev_layer = maxpool
        unwrap = SeqUnwrapper(250)
        unwrap.connect(prev_layer, y_time, y_seq_id)
        prev_layer = unwrap
        rng = np.random.RandomState(23455)
        conv = LeNetConvPoolLayer()
        conv.connect(prev_layer, rng, (20, 1, 5, emb_size), (8, 1, ))
        prev_layer = conv

        logging.info('Conv output size: %d' % conv.size)

        costs = []
        predictions = []
        for slot in slots:
            logging.info('Building output classifier for %s.' % slot)
            n_classes = len(slot_classes[slot])
            slot_mlp = MLP([oclf_n_hidden  ] * oclf_n_layers + [n_classes],
                           [oclf_activation] * oclf_n_layers + ['softmax'],
                           [p_drop         ] * oclf_n_layers + [0.0      ],
                           name="mlp_%s" % slot)
            slot_mlp.connect(prev_layer)
            predictions.append(slot_mlp.output(dropout_active=False))

            slot_objective = CrossEntropyObjective()
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
        clipnorm = 0.5
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

    def prepare_data_train(self, seqs, slots):
        return self._prepare_data(seqs, slots, with_labels=True)

    def prepare_data_predict(self, seqs, slots):
        return self._prepare_data(seqs, slots, with_labels=False)

    def _prepare_y_token_labels_padding(self):
        token_padding = []
        for slot in self.slots:
            token_padding.append(0)
            token_padding.append(0)

        return [token_padding]

    def _prepare_data(self, seqs, slots, with_labels=True):
        x = []
        x_score = []
        x_actor = []
        y_seq_id = []
        y_time = []
        y_labels = [[] for slot in slots]
        y_weights = []
        for item in seqs:
            x.append(item['data'])
            x_score.append(item['data_score'])
            x_actor.append(item['data_actor'])

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

        x_score = padded(x_score).transpose(1, 0)
        x_score = np.array(x_score, dtype=np.float32)[:,:]
        x_score = (x_score / 10.0)

        data = [x]
        if self.x_include_score:
            data.append(x_score)
        data.extend([y_seq_id, y_time])
        if with_labels:
            data.extend(y_labels)

        return tuple(data)





