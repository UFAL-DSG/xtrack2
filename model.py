import inspect
import logging
import time

import theano
import theano.tensor as tt

from passage import updates
from passage.iterators import padded
from passage.layers import *
from passage.model import NeuralModel


class Model(NeuralModel):
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
        args = Model.__init__.func_code.co_varnames[:Model.__init__.func_code.co_argcount]
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
        if init_emb_from:
            input_token_layer.init_from(init_emb_from, vocab)
            logging.info('Initializing token embeddings from: %s'
                         % init_emb_from)
        else:
            logging.info('Initializing token embedding randomly.')
        self.input_emb = input_token_layer.wv

        prev_layer = input_token_layer



        input_layers = [
             input_token_layer
        ]
        if x_include_score:
            x_score = tt.imatrix()
            input_score_layer = Embedding(name="emb_score",
                                          size=emb_size,
                                          n_features=n_input_score_bins,
                                          input=x_score)
            input_layers.append(input_score_layer)

            input_args.append(x_score)

        if x_include_token_ftrs:
            token_n_features = len(token_features.values()[0])
            input_token_features_layer = Embedding(name="emb_ftr",
                                                   size=token_n_features,
                                                   n_features=n_input_tokens,
                                                   input=x,
                                                   static=True)
            input_token_features_layer.init_from_dict(token_features)

            ftrs_to_emb = Dense(name='ftr2emb',
                                size=emb_size,
                                activation='linear')
                                # FIX: p_drop=p_drop)
            ftrs_to_emb.connect(input_token_features_layer)
            input_layers.append(ftrs_to_emb)

        sum_layer = SumLayer(layers=input_layers)
        prev_layer = sum_layer

        if input_n_layers > 0:
            input_transform = MLP([input_n_hidden  ] * input_n_layers,
                                  [input_activation] * input_n_layers,
                                  p_drop=p_drop)
            input_transform.connect(prev_layer)
            prev_layer = input_transform

        if token_supervision:
            slot_value_pred = MLP([len(slots) * 2], ['sigmoid'],
                                  p_drop=[p_drop], name='ts')
            slot_value_pred.connect(prev_layer)

            y_tokens_label = tt.itensor3()
            token_supervision_loss_layer = TokenSupervisionLossLayer()
            token_supervision_loss_layer.connect(slot_value_pred, y_tokens_label)

            if debug:
                self._token_supervision_loss = theano.function(input_args + [
                    y_tokens_label], token_supervision_loss_layer.output())

        else:
            token_supervision_loss_layer = None
            y_tokens_label = None




        logging.info('There are %d input layers.' % input_n_layers)

        if debug:
            self._lstm_input = theano.function(input_args, prev_layer.output())

        h_t_layer = IdentityInput(None, n_cells)
        mlps = []
        mlp_params = []
        for slot in slots:
            n_classes = len(slot_classes[slot])
            slot_mlp = MLP([oclf_n_hidden  ] * oclf_n_layers + [n_classes],
                           [oclf_activation] * oclf_n_layers + ['softmax'],
                           [0.0         ] * oclf_n_layers + [0.0      ],
                           name="mlp_%s" % slot)
            slot_mlp.connect(h_t_layer)
            mlps.append(slot_mlp)
            mlp_params.extend(slot_mlp.get_params())


        for i in range(rnn_n_layers):
            # Forward LSTM layer.
            logging.info('Creating LSTM layer with %d neurons.' % (n_cells))
            if x_include_mlp:
                f_lstm_layer = LstmWithMLP(name="flstm_%d" % i,
                                       size=n_cells,
                                       seq_output=True,
                                       out_cells=False,
                                       peepholes=lstm_peepholes,
                                       p_drop=p_drop,
                                       enable_branch_exp=enable_branch_exp,
                                       mlps=mlps)
            else:
                f_lstm_layer = LstmRecurrent(name="flstm_%d" % i,
                                       size=n_cells,
                                       seq_output=True,
                                       out_cells=False,
                                       peepholes=lstm_peepholes,
                                       p_drop=p_drop,
                                       enable_branch_exp=enable_branch_exp
                )

            f_lstm_layer.connect(prev_layer)





            if lstm_bidi:
                b_lstm_layer = LstmRecurrent(name="blstm_%d" % i,
                                       size=n_cells,
                                       seq_output=True,
                                       out_cells=False,
                                       backward=True,
                                       peepholes=lstm_peepholes,
                                       p_drop=p_drop,
                                       enable_branch_exp=enable_branch_exp)
                b_lstm_layer.connect(prev_layer)

                lstm_zip = ZipLayer(concat_axis=2, layers=[f_lstm_layer,
                                                         b_lstm_layer])
                prev_layer = lstm_zip
                if debug:
                    self._lstm_output = theano.function(input_args,
                                                   [prev_layer.output(),
                                                    f_lstm_layer.output(),
                                                    b_lstm_layer.output()])
            else:

                if debug:
                    self._lstm_output = theano.function(input_args,
                                                        [prev_layer.output(),
                                                         f_lstm_layer.output()])

                prev_layer = f_lstm_layer

        assert prev_layer is not None

        y_seq_id = tt.ivector()
        y_time = tt.ivector()
        y_weight = tt.vector()
        y_label = {}
        for slot in slots:
            y_label[slot] = tt.ivector(name='y_label_%s' % slot)

        cpt = CherryPick()
        cpt.connect(prev_layer, y_time, y_seq_id)

        costs = []
        predictions = []
        for slot, slot_lstm_mlp in zip(slots, mlps):
            logging.info('Building output classifier for %s.' % slot)
            n_classes = len(slot_classes[slot])
            slot_mlp = MLP([oclf_n_hidden  ] * oclf_n_layers + [n_classes],
                           [oclf_activation] * oclf_n_layers + ['softmax'],
                           [p_drop         ] * oclf_n_layers + [0.0      ],
                           name="mlp_%s" % slot, init=inits.copy(mlp_params))
            slot_mlp.connect(cpt)
            predictions.append(slot_mlp.output(dropout_active=False))

            slot_objective = WeightedCrossEntropyObjective()
            slot_objective.connect(
                y_hat_layer=slot_mlp,
                y_true=y_label[slot],
                y_weights=y_weight
            )
            costs.append(slot_objective)
        if token_supervision:
            costs.append(token_supervision_loss_layer)

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
        loss_args += [y_weight]
        loss_args += [y_label[slot] for slot in slots]
        if token_supervision:
            loss_args += [y_tokens_label]

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

    def init_word_embeddings(self, w):
        self.input_emb.set_value(w)

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
        x_actor = padded(x_actor, is_int=True).transpose(1, 0)

        x_score = np.array(x_score, dtype=np.int32)[:,:]

        y_weights = np.array(y_weights, dtype=np.float32)

        y_token_labels_padding = self._prepare_y_token_labels_padding()

        data = [x]
        if self.x_include_score:
            data.append(x_score)
        data.extend([y_seq_id, y_time])
        if with_labels:
            data.append(y_weights)
            data.extend(y_labels)

        return tuple(data)





