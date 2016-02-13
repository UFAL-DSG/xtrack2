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
from passage.lstm_with_mlp import LstmWithMLP


class Model(NeuralModel):
    def _log_classes_info(self):
        for slot, vals in self.slot_classes.iteritems():
            logging.info('  %s:' % slot)
            for val, val_ndx in sorted(vals.iteritems(), key=lambda x: x[1]):
                logging.info('    - %s (%d)' % (val, val_ndx))

    def __init__(self, slots, slot_classes, emb_size, no_train_emb,
                 x_include_score, x_include_token_ftrs, x_include_mlp,
                 split_cost,
                 wcn_aggreg,
                 n_input_tokens, n_input_score_bins, n_cells,
                 rnn_n_layers,
                 lstm_type, lstm_update_thresh, lstm_peepholes, lstm_bidi, opt_type,
                 oclf_n_hidden, oclf_n_layers, oclf_activation,
                 debug, p_drop,
                 init_emb_from, vocab, vocab_ftr_map, ftr_emb_size,
                 input_n_layers, input_n_hidden, input_activation,
                 token_features, token_supervision, use_loss_mask,
                 momentum, enable_branch_exp, l1, l2, build_train=True, x_include_orig=False):
        self.store_init_args(locals())

        self.vocab = vocab

        self.slots = slots
        self.slot_classes = slot_classes

        #
        # Input
        #

        x = T.itensor3(name='x')
        x_tagged = T.itensor3(name='x_tagged')
        x_conf = T.tensor3(name='x_conf')

        input_args = []
        input_zip_layers = []
        input_args.append(x)
        input_args.append(x_tagged)
        input_args.append(x_conf)
        input_token_layer = Embedding(name="emb",
                                      size=emb_size,
                                      n_features=n_input_tokens,
                                      input=x[:, :, 0],
                                      static=no_train_emb)
        tagged_input_token_layer = Embedding(name="taggedemb",
                                      size=emb_size,
                                      n_features=n_input_tokens,
                                      input=x_tagged[:, :, 0],
                                      static=no_train_emb)
        tagged_input_token_layer.wv = input_token_layer.wv

        input_conf_layer = IdentityInput(x_conf[:, :, 0, np.newaxis], 1)

        input_zip_layers.append(input_token_layer)
        input_zip_layers.append(tagged_input_token_layer)
        input_zip_layers.append(input_conf_layer)

        if init_emb_from and (init_emb_from.endswith('.txt') or init_emb_from.endswith('.gz')):
            input_token_layer.init_from(init_emb_from, vocab)

        prev_layer = ZipLayer(2, input_zip_layers)

        #
        # Input preprocessing.
        #

        rev_flat = ReshapeLayer(x.shape[0] * x.shape[1], prev_layer.size)
        rev_flat.connect(prev_layer)
        prev_layer = rev_flat

        input_mlp_layer = MLP([input_n_hidden  ] * input_n_layers,
                              [input_activation] * input_n_layers,
                              [p_drop          ] * input_n_layers,
                           name="input_mlp")
        input_mlp_layer.connect(prev_layer)
        prev_layer = input_mlp_layer

        reshape_back = ReshapeLayer(x.shape[0], x.shape[1], input_n_hidden)
        reshape_back.connect(prev_layer)
        prev_layer = reshape_back

        if debug:
            self._lstm_input = theano.function(input_args, prev_layer.output())

        #
        # Sequential processing with LSTM
        #

        n_classes = len(slot_classes['food'])
        lstm_layer = LstmWithMLP(name="lstm",
                                 size=n_cells,
                                 seq_output=True,
                                 out_cells=False,
                                 peepholes=lstm_peepholes,
                                 p_drop=p_drop,
                                 enable_branch_exp=enable_branch_exp,
                                 mlp_n_classes=n_classes,
                                 mlp_n_hidden=oclf_n_hidden,
                                 mlp_n_layers=oclf_n_layers)
        lstm_layer.connect(prev_layer)
        prev_layer = lstm_layer


        y_seq_id = tt.ivector()
        y_time = tt.ivector()

        y_label = {}
        for slot in slots:
            y_label[slot] = tt.ivector(name='y_label_%s' % slot)

        cpt = CherryPick()
        cpt.connect(prev_layer, y_time, y_seq_id)

        costs = []
        predictions = []
        slot_objective = CrossEntropyObjective()
        slot_objective.connect(
            y_hat_layer=cpt,
            y_true=y_label['food']
        )
        costs.append(slot_objective)
        predictions.append(cpt.output(dropout_active=False))

        cost = SumOut()
        cost.connect(*costs)  #, scale=1.0 / len(slots))
        self.params = params = list(cost.get_params())
        n_params = sum(p.get_value().size for p in params)
        logging.info('This model has %d parameters:' % n_params)
        for param in sorted(params, key=lambda x: x.name):
            logging.info('  - %20s: %10d' % (param.name, param.get_value(

            ).size, ))



        lr = tt.scalar('lr')
        clip = 5.0
        reg = updates.Regularizer(l1=l1, l2=l2)
        if opt_type == "rprop":
            updater = updates.RProp(lr=lr, clipnorm=clipnorm)
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

        self._train = []

        train_obj = []
        if not split_cost:
            train_obj.append(('joint', cost, [y_label[slot] for slot in slots]))
        else:
            for slot, slot_cost in zip(slots, costs):
                train_obj.append((slot, slot_cost, [y_label[slot]]))

        loss_args = list(input_args)
        loss_args += [y_seq_id, y_time]
        loss_args += [y_label[slot] for slot in slots]

        for obj_name, obj, y_label_args in train_obj:
            cost_value = obj.output(dropout_active=True)
            obj_params = list(obj.get_params())

            if build_train:
                model_updates = updater.get_updates(obj_params, cost_value)

                train_args = [lr] + loss_args
                update_ratio = updater.get_update_ratio(obj_params, model_updates)

                logging.info('Preparing %s train function for %s.' % (opt_type, obj_name, ))
                t = time.time()
                self._train.append(theano.function(train_args, [cost_value / y_time.shape[0], update_ratio],
                                              updates=model_updates, on_unused_input='ignore'))
                logging.info('Preparation done. Took: %.1f' % (time.time() - t))

        self._loss = theano.function(loss_args, cost.output(dropout_active=True))

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

    def _get_token(self, vocab_rev, word):
        word_str = vocab_rev[word]
        res = 0

        for slot in self.slots:
            if word_str == slot:
                res |= 1
            for val in self.slot_classes[slot].values():
                if val == word_str:
                    res |= 2

        return res

    def _prepare_data(self, seqs, slots, with_labels=True, debug_data=False, dense_labels=False):
        vocab_rev = {v: k for k, v in self.vocab.iteritems()}

        x = []
        x_orig = []
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
            data_orig = []
            data_score = []
            #data_ftrs = []
            #assert len(item['data']) == len(item['ftrs'])
            assert len(item['data']) == len(item['data_score'])

            if not 'data_orig' in item:
                item['data_orig'] = item['data']
                #logging.warning("Replacing data_orig with data because data_orig does not exist.")

            assert len(item['data']) == len(item['data_orig'])

            for words, words_orig, score in zip(item['data'], item['data_orig'], item['data_score']): #, item['ftrs']):
                new_words = []
                new_words_orig = []
                new_score = []
                #new_ftrs = []
                assert type(words) is list
                for word, word_orig, word_score in sorted(zip(words, words_orig, score), key=lambda (w, wo, s, ): -s)[:wcn_cnt]:
                    new_words.append(word)
                    word_orig = self._get_token(vocab_rev, word)

                    new_words_orig.append(word_orig)
                    new_score.append(np.exp(word_score))

                n_missing = max(0, wcn_cnt - len(words))
                new_words.extend(n_missing * [0])
                new_words_orig.extend(n_missing * [0])
                new_score.extend(n_missing * [1.0])
                #new_ftrs.extend(n_missing * [[0] * len(ftrs[0])])

                data.append(new_words)
                data_orig.append(new_words_orig)
                data_score.append(new_score)
                #data_ftrs.append(new_ftrs)

            #import ipdb; ipdb.set_trace()

            x.append(data)
            x_orig.append(data_orig)
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
        x_orig = padded(x_orig, is_int=True, pad_by=[[0] * wcn_cnt]).transpose(1, 0, 2)
        x_score = padded(x_score, pad_by=[[0.0] * wcn_cnt]).transpose(1, 0, 2)
        #x_ftrs = padded(x_ftrs, pad_by=[[[0.0] * len(x_ftrs[0][0][0])] * wcn_cnt]).transpose(1, 0, 2, 3) #[:, :, 0]

        if debug_data:
            import ipdb; ipdb.set_trace()

        data = []
        data.append(x_orig)
        data.append(x)

        #if self.x_include_score:
        data.append(x_score)
        #data.append(x_ftrs)

        data.extend([y_seq_id, y_time])
        if with_labels:
            #data.append(y_masks)
            data.extend(y_labels)

        return tuple(data)





