import inspect
import logging
import time

import theano
import theano.tensor as tt

from passage import updates
from passage.iterators import padded
from passage.layers import *
from passage.lstm_with_confidence import LstmWithConfidence
from passage.model import NeuralModel


class Model(NeuralModel):
    def _log_classes_info(self):
        for slot, vals in self.slot_classes.iteritems():
            logging.info('  %s:' % slot)
            for val, val_ndx in sorted(vals.iteritems(), key=lambda x: x[1]):
                logging.info('    - %s (%d)' % (val, val_ndx))

    def __init__(self, slots, slot_classes, emb_size, no_train_emb,
                 x_include_score, x_include_token_ftrs, x_include_mlp,
                 wcn_aggreg,
                 n_input_tokens, n_input_score_bins, n_cells,
                 rnn_n_layers,
                 lstm_type, lstm_update_thresh, lstm_peepholes, lstm_bidi, opt_type,
                 oclf_n_hidden, oclf_n_layers, oclf_activation,
                 debug, p_drop,
                 init_emb_from, vocab, vocab_ftr_map, ftr_emb_size,
                 input_n_layers, input_n_hidden, input_activation,
                 token_features, token_supervision, use_loss_mask,
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

        x = T.itensor3(name='x')
        input_args = []
        input_zip_layers = []
        input_args.append(x)
        input_token_layer = Embedding(name="emb",
                                      size=emb_size,
                                      n_features=n_input_tokens,
                                      input=x,
                                      static=no_train_emb)
        input_zip_layers.append(input_token_layer)

        if token_features:
            input_ftrs_layer = FeatureEmbedding(name="ftremb",
                                                size=ftr_emb_size,
                                                vocab=vocab,
                                                vocab_ftr_map=vocab_ftr_map,
                                                input=x)
            input_zip_layers.append(input_ftrs_layer)

        x_conf = T.tensor3(name='x_conf')
        input_args.append(x_conf)
        input_conf_layer = IdentityInput(x_conf[:, :, :, np.newaxis], 1)
        input_zip_layers.append(input_conf_layer)

        prev_layer = ZipLayer(3, input_zip_layers)

        rev_flat =  ReshapeLayer(x.shape[0] * x.shape[1] * x.shape[2], prev_layer.size)
        rev_flat.connect(prev_layer)
        prev_layer = rev_flat

        input_mlp_layer = MLP([input_n_hidden  ] * input_n_layers,
                              [input_activation] * input_n_layers,
                              [p_drop          ] * input_n_layers,
                           name="input_mlp")
        input_mlp_layer.connect(prev_layer)
        prev_layer = input_mlp_layer

        reshape_back = ReshapeLayer(x.shape[0], x.shape[1], x.shape[2], input_n_hidden)
        reshape_back.connect(prev_layer)
        prev_layer = reshape_back

        #logging.info("reshape_back size %d" % reshape_back.size)

        if wcn_aggreg == 'flatten':
            wcn_aggreg_layer = FlattenLayer(3, 5)
            wcn_aggreg_layer.connect(prev_layer)
            prev_layer = wcn_aggreg_layer

            wcn_out = Dense(name="wcn_out", size=emb_size, activation=input_activation)
            wcn_out.connect(prev_layer)
            prev_layer = wcn_out
        elif wcn_aggreg == 'max':
            wcn_aggreg_layer = MaxPooling("wcn_max", pool_dimension=2)
            wcn_aggreg_layer.connect(prev_layer)
            prev_layer = wcn_aggreg_layer
        else:
            assert False, "Unknown wcn_aggreg: %s" % wcn_aggreg

        logging.info("wcn_aggreg size %d" % prev_layer.size)

        #flat_wcn_out = ReshapeLayer(x.shape[0] * x.shape[1], 5 * input_n_hidden)
        #flat_wcn_out.connect(prev_layer)
        #prev_layer = flat_wcn_out



        #flat_wcn_out_rev = ReshapeLayer(x.shape[0], x.shape[1], emb_size)
        #flat_wcn_out_rev.connect(prev_layer)
        #prev_layer = flat_wcn_out_rev

        logging.info('There are %d input layers.' % input_n_layers)

        if debug:
            self._lstm_input = theano.function(input_args, prev_layer.output())

        # Forward LSTM layer.
        assert rnn_n_layers > 0
        for i in range(rnn_n_layers):
            logging.info('Creating LSTM layer with %d neurons.' % (n_cells))
            lstm_args = dict(name="lstm%d" % i,
                                   size=n_cells,
                                   seq_output=True,
                                   out_cells=False,
                                   peepholes=lstm_peepholes,
                                   p_drop=p_drop,
                                   enable_branch_exp=enable_branch_exp)
            lstm_connect_args = [prev_layer]

            if lstm_type == 'ngram':
                lstm_cls = NGramLSTM
            elif lstm_type == 'vanilla':
                lstm_cls = LstmRecurrent
            elif lstm_type == 'with_conf':
                lstm_cls = LstmWithConfidence
                lstm_args.update(update_thresh=lstm_update_thresh)
                lstm_connect_args.append(x_ftrs)
            else:
                raise Exception('Unknown LSTM type: %s' % lstm_type)

            lstm_layer = lstm_cls(**lstm_args)
            lstm_layer.connect(*lstm_connect_args)

            prev_layer = lstm_layer

        if debug:
            self._lstm_output = theano.function(input_args,
                                                [prev_layer.output(),
                                                 lstm_layer.output()])

        #prev_layer = lstm_layer

        assert prev_layer is not None

        y_seq_id = tt.ivector()
        y_time = tt.ivector()
        y_masks = tt.imatrix()
        y_label = {}
        for slot in slots:
            y_label[slot] = tt.ivector(name='y_label_%s' % slot)

        cpt = CherryPick()
        cpt.connect(prev_layer, y_time, y_seq_id)

        if not use_loss_mask:
            y_masks = y_masks * 0 + 1

        costs = []
        predictions = []
        for i, slot in enumerate(slots):
            logging.info('Building output classifier for %s.' % slot)
            n_classes = len(slot_classes[slot])
            slot_mlp = MLP([oclf_n_hidden  ] * oclf_n_layers + [n_classes],
                           [oclf_activation] * oclf_n_layers + ['softmax'],
                           [p_drop         ] * oclf_n_layers + [0.0      ],
                           name="mlp_%s" % slot)
            slot_mlp.connect(cpt)

            pred = slot_mlp.output(dropout_active=False)
            predictions.append(pred)

            slot_objective = WeightedCrossEntropyObjective()
            slot_objective.connect(
                y_hat_layer=slot_mlp,
                y_true=y_label[slot],
                y_weights=y_masks[:, i]
            )
            costs.append(slot_objective)

            logging.info('Creating understanding function.')

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
        clip = 5.0
        reg = updates.Regularizer(l1=l1, l2=l2)
        if opt_type == "rprop":
            updater = updates.RProp(lr=lr, clipnorm=clipnorm)
            model_updates = updater.get_updates(params, cost_value)
        elif opt_type == "sgd":
            updater = updates.SGD(lr=lr, clipnorm=clip, regularizer=reg)
        elif opt_type == "rmsprop":
            updater = updates.RMSprop(lr=lr, clipnorm=clipnorm, regularizer=reg)  #, regularizer=reg)
        elif opt_type == "adam":
            #reg = updates.Regularizer(maxnorm=5.0)
            updater = updates.Adam(lr=lr, clip=clip, regularizer=reg)  #,
            # regularizer=reg)
        elif opt_type == "momentum":
            updater = updates.Momentum(lr=lr, momentum=momentum, clipnorm=clipnorm, regularizer=reg)
        else:
            raise Exception("Unknonw opt.")

        loss_args = list(input_args)
        loss_args += [y_seq_id, y_time, y_masks]
        loss_args += [y_label[slot] for slot in slots]

        if build_train:
            model_updates = updater.get_updates(params, cost_value)

            train_args = [lr] + loss_args
            update_ratio = updater.get_update_ratio(params, model_updates)

            logging.info('Preparing %s train function.' % opt_type)
            t = time.time()
            self._train = theano.function(train_args, [cost_value / y_time.shape[0], update_ratio],
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

    def prepare_data_train(self, seqs, slots, debug_data=False, dense_labels=False):
        return self._prepare_data(seqs, slots, with_labels=True, debug_data=debug_data, dense_labels=dense_labels)

    def prepare_data_predict(self, seqs):
        return self._prepare_data(seqs, [], with_labels=False)

    def _prepare_y_token_labels_padding(self):
        token_padding = []
        for slot in self.slots:
            token_padding.append(0)
            token_padding.append(0)

        return [token_padding]

    def _prepare_data(self, seqs, slots, with_labels=True, debug_data=False, dense_labels=False):
        x = []
        x_score = []
        #x_ftrs = []
        y_seq_id = []
        y_time = []
        y_labels = [[] for slot in slots]
        y_weights = []
        y_masks = []
        wcn_cnt = 5
        for item in seqs:
            data = []
            data_score = []
            #data_ftrs = []
            #assert len(item['data']) == len(item['ftrs'])
            assert len(item['data']) == len(item['data_score'])
            for words, score in zip(item['data'], item['data_score']): #, item['ftrs']):
                new_words = []
                new_score = []
                #new_ftrs = []
                assert type(words) is list
                for word, word_score in sorted(zip(words, score), key=lambda (w, s, ): -s)[:wcn_cnt]:
                    new_words.append(word)
                    new_score.append(np.exp(word_score))
                    #new_ftrs.append(word_ftrs)

                n_missing = max(0, wcn_cnt - len(words))
                new_words.extend(n_missing * [0])
                new_score.extend(n_missing * [1.0])
                #new_ftrs.extend(n_missing * [[0] * len(ftrs[0])])

                data.append(new_words)
                data_score.append(new_score)
                #data_ftrs.append(new_ftrs)

            #import ipdb; ipdb.set_trace()

            x.append(data)
            x_score.append(data_score)
            #x_ftrs.append(data_ftrs)

            labels = item['labels']

            prev_y_time = 0
            for label in labels:
                if dense_labels:
                    for i in range(prev_y_time, label['time']):
                        y_seq_id.append(len(x) - 1)
                        y_seq_id.append(len(x) - 1)
                        y_time.append(i)
                        y_time.append(i)
                        for i, slot in enumerate(slots):
                            if prev_y_time == 0:
                                y_labels[i].append(0)
                            else:
                                y_labels[i].append(y_labels[i][-1])

                            lbl_val = label['slots'][slot]
                            if lbl_val < 0:
                                lbl_val = len(self.slot_classes[slot]) + lbl_val
                            y_labels[i].append(lbl_val)

                    prev_y_time = label['time'] + 1

                y_seq_id.append(len(x) - 1)
                y_time.append(label['time'])

                for i, slot in enumerate(slots):
                    lbl_val = label['slots'][slot]
                    if lbl_val < 0:
                        lbl_val = len(self.slot_classes[slot]) + lbl_val
                    y_labels[i].append(lbl_val)
                y_weights.append(label['score'])

                y_mask = []
                for i, slot in enumerate(slots):
                    if slot in label['slots_mentioned']:
                        y_mask.append(1)
                    else:
                        y_mask.append(0)
                y_masks.append(y_mask)


        x = padded(x, is_int=True, pad_by=[[0] * wcn_cnt]).transpose(1, 0, 2)
        x_score = padded(x_score, pad_by=[[0.0] * wcn_cnt]).transpose(1, 0, 2)
        #x_ftrs = padded(x_ftrs, pad_by=[[[0.0] * len(x_ftrs[0][0][0])] * wcn_cnt]).transpose(1, 0, 2, 3) #[:, :, 0]

        if debug_data:
            import ipdb; ipdb.set_trace()

        data = [x]
        #if self.x_include_score:
        data.append(x_score)
        #data.append(x_ftrs)

        data.extend([y_seq_id, y_time])
        if with_labels:
            data.append(y_masks)
            data.extend(y_labels)

        return tuple(data)





