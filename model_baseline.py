import inspect
import logging
import time

import theano
import theano.tensor as tt

from passage import updates
from passage.iterators import padded
from passage.layers import *
from passage.model import NeuralModel


class BaselineModel(NeuralModel):
    def _log_classes_info(self):
        for slot, vals in self.slot_classes.iteritems():
            logging.info('  %s:' % slot)
            for val, val_ndx in sorted(vals.iteritems(), key=lambda x: x[1]):
                logging.info('    - %s (%d)' % (val, val_ndx))

    def __init__(self, slots, slot_classes, opt_type,
                 oclf_n_hidden, oclf_n_layers, oclf_activation,
                 n_cells,
                 debug, p_drop,
                 vocab,
                 input_n_layers, input_n_hidden, input_activation,
                 token_features, token_supervision,
                 momentum, enable_branch_exp, l1, l2, build_train=True):
        args = BaselineModel.__init__.func_code.co_varnames[
               :BaselineModel.__init__.func_code.co_argcount]
        self.init_args = {}
        for arg in args:
            if arg != 'self':
                self.init_args[arg] = locals()[arg]

        self.vocab = vocab

        self.slots = slots
        self.slot_classes = slot_classes


        logging.info('We have the following classes:')
        self._log_classes_info()

        x = T.tensor3()
        input_args = [x]
        input_layer = IdentityInput(x, len(self.vocab))

        prev_layer = input_layer

        if input_n_layers > 0:
            input_transform = MLP([input_n_hidden  ] * input_n_layers,
                                  [input_activation] * input_n_layers,
                                  p_drop=p_drop)
            input_transform.connect(prev_layer)
            prev_layer = input_transform

        logging.info('There are %d input layers.' % input_n_layers)

        if debug:
            self._lstm_input = theano.function(input_args, prev_layer.output())


        logging.info('Creating LSTM layer with %d neurons.' % (n_cells))
        f_lstm_layer = LstmRecurrent(name="lstm",
                               size=n_cells,
                               seq_output=True,
                               out_cells=False,
                               peepholes=False,
                               p_drop=p_drop,
                               enable_branch_exp=enable_branch_exp
        )
        f_lstm_layer.connect(prev_layer)

        prev_layer = f_lstm_layer

        y_seq_id = tt.ivector()
        y_time = tt.ivector()
        y_label = {}
        for slot in slots:
            y_label[slot] = tt.ivector(name='y_label_%s' % slot)

        cpt = CherryPick()
        cpt.connect(prev_layer, y_time, y_seq_id)

        costs = []
        predictions = []
        for slot in slots:
            logging.info('Building output classifier for %s.' % slot)
            n_classes = len(slot_classes[slot])
            slot_mlp = MLP([oclf_n_hidden  ] * oclf_n_layers + [n_classes],
                           [oclf_activation] * oclf_n_layers + ['softmax'],
                           [p_drop         ] * oclf_n_layers + [0.0      ],
                           name="mlp_%s" % slot)
            slot_mlp.connect(cpt)
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

        lr = tt.scalar('lr')
        clipnorm = 0.5
        reg = updates.Regularizer(l1=l1, l2=l2)
        if opt_type == "rprop":
            updater = updates.RProp(lr=lr, clipnorm=clipnorm)
            model_updates = updater.get_updates(params, cost_value)
        elif opt_type == "sgd":
            updater = updates.SGD(lr=lr, clipnorm=clipnorm, regularizer=reg)
        elif opt_type == "rmsprop":
            updater = updates.RMSprop(lr=lr, clipnorm=clipnorm, regularizer=reg)  #, regularizer=reg)
        elif opt_type == "adam":
            #reg = updates.Regularizer(maxnorm=5.0)
            updater = updates.Adam(lr=lr, clipnorm=clipnorm, regularizer=reg)  #,
            # regularizer=reg)
        elif opt_type == "momentum":
            updater = updates.Momentum(lr=lr, momentum=momentum, clipnorm=clipnorm, regularizer=reg)
        else:
            raise Exception("Unknonw opt.")

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

    def init_loaded(self):
        pass

    def prepare_data_train(self, seqs, slots):
        return self._prepare_data(seqs, slots, with_labels=True)

    def prepare_data_predict(self, seqs):
        return self._prepare_data(seqs, [], with_labels=False)

    def _prepare_y_token_labels_padding(self):
        token_padding = []
        for slot in self.slots:
            token_padding.append(0)
            token_padding.append(0)

        return [token_padding]

    def _prepare_data(self, seqs, slots, with_labels=True):
        x = []
        y_seq_id = []
        y_time = []
        y_labels = [[] for slot in slots]
        for item in seqs:
            x_vecs = []
            for features in item['data']:
                x_vec = np.zeros((len(self.vocab), ))
                for ftr, val in features.iteritems():
                    x_vec[self.vocab[ftr]] = val
                x_vecs.append(x_vec)

            x.append(x_vecs)

            labels = item['labels']

            for label in labels:
                y_seq_id.append(len(x) - 1)
                y_time.append(label['time'])

                for i, slot in enumerate(slots):
                    lbl_val = label['slots'][slot]
                    if lbl_val < 0:
                        lbl_val = len(self.slot_classes[slot]) + lbl_val
                    y_labels[i].append(lbl_val)

        x_zero_pad = np.zeros((len(self.vocab), ))
        x = padded(x, pad_by=[x_zero_pad]).transpose(1, 0, 2)

        data = [x]
        data.extend([y_seq_id, y_time])
        if with_labels:
            data.extend(y_labels)

        return tuple(data)





