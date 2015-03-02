import logging
import time

import theano
import theano.tensor as tt

from passage import updates
from passage.iterators import padded
from passage.layers import *
from passage.model import NeuralModel

class Model(NeuralModel):
    def __init__(self, slots, slot_classes, emb_size, disable_input_ftrs,
                 n_input_tokens, n_cells,
                 rnn_type, rnn_n_layers,
                 lstm_no_peepholes, opt_type,
                 oclf_n_hidden, oclf_n_layers, oclf_activation,
                 debug, p_drop,
                 init_emb_from, vocab,
                 input_n_layers, input_n_hidden, input_activation,
                 token_features,
                 momentum):
        self.vocab = vocab
        self.slots = slots
        self.slot_classes = slot_classes

        x = T.imatrix()

        input_token_layer = Embedding(name="emb",
                                      size=emb_size,
                                      n_features=n_input_tokens,
                                      input=x)
        if init_emb_from:
            input_token_layer.init_from(init_emb_from, vocab)
            logging.info('Initializing embeddings from: %s' % init_emb_from)
        else:
            logging.info('Initializing embedding randomly.')
        self.input_emb = input_token_layer.wv

        prev_layer = input_token_layer

        x_score = tt.tensor3()
        input_score_layer = IdentityInput(x_score, 1)

        x_switch = tt.itensor3()
        input_switch_layer = IdentityInput(x_switch, 1)

        zip_layers = [
             input_token_layer,
             input_score_layer,
             input_switch_layer,
        ]

        if not disable_input_ftrs:
            token_n_features = len(token_features.values()[0])
            input_token_features_layer = Embedding(name="emb_ftr",
                                                   size=token_n_features,
                                                   n_features=n_input_tokens,
                                                   input=x,
                                                   static=True)
            input_token_features_layer.init_from_dict(token_features)
            zip_layers.append(input_token_features_layer)

        zip_layer = ZipLayer(concat_axis=2,
                             layers=zip_layers)
        prev_layer = zip_layer

        if input_n_layers > 0:
            input_transform = MLP([input_n_hidden  ] * input_n_layers,
                                  [input_activation] * input_n_layers,
                                  p_drop=p_drop)
            input_transform.connect(prev_layer)
            prev_layer = input_transform

        logging.info('There are %d input layers.' % input_n_layers)

        if debug:
            self._lstm_input = theano.function([x, x_score, x_switch],
                                               prev_layer.output())

        lstm_layer = None
        for i in range(rnn_n_layers):
            logging.info('Creating RNN layer: %s with %d neurons.' % (
                rnn_type, n_cells))
            if rnn_type == 'lstm':
                lstm_layer = LstmRecurrent(name="lstm",
                                       size=n_cells,
                                       seq_output=True,
                                       out_cells=False,
                                       peepholes=not lstm_no_peepholes,
                                       p_drop=p_drop)

            elif rnn_type == 'rnn':
                lstm_layer = Recurrent(name="lstm",
                                       size=n_cells,
                                       seq_output=True,
                                       p_drop=p_drop)
            else:
                raise Exception('Unknown RNN type.')
            lstm_layer.connect(prev_layer)
            prev_layer = lstm_layer

        assert lstm_layer is not None

        y_seq_id = tt.ivector()
        y_time = tt.ivector()
        y_label = {}
        for slot in slots:
            y_label[slot] = tt.ivector(name='y_label_%s' % slot)

        cpt = CherryPick()
        cpt.connect(lstm_layer, y_time, y_seq_id)

        costs = []
        predictions = []
        for slot in slots:
            logging.info('Building output classifier for %s.' % slot)
            n_classes = len(slot_classes[slot])
            slot_mlp = MLP([oclf_n_hidden  ] * oclf_n_layers + [n_classes],
                           [oclf_activation] * oclf_n_layers + ['softmax'],
                           name="mlp_%s" % slot, p_drop=p_drop)
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
        logging.info('This model has %d parameters.' % n_params)

        cost_value = cost.output(dropout_active=False)
        assert p_drop == 0.0

        lr = tt.scalar('lr')
        if opt_type == "rprop":
            updater = updates.RProp(lr=lr, clipnorm=5.0)
            model_updates = updater.get_updates(params, cost_value)
        elif opt_type == "sgd":
            updater = updates.SGD(lr=lr, clipnorm=5.0)
        elif opt_type == "rmsprop":
            #reg = updates.Regularizer(maxnorm=5.0)
            updater = updates.RMSprop(lr=lr, clipnorm=5.0)  #, regularizer=reg)
        elif opt_type == "adam":
            #reg = updates.Regularizer(maxnorm=5.0)
            updater = updates.Adam(lr=lr, b1=0.01, b2=0.01, clipnorm=5.0)  #,
            # regularizer=reg)
        elif opt_type == "momentum":
            updater = updates.Momentum(lr=lr, momentum=momentum, clipnorm=5.0)
        else:
            raise Exception("Unknonw opt.")

        model_updates = updater.get_updates(params, cost_value)

        input_args = [x]
        input_args += [x_score, x_switch]
        input_args += [y_seq_id, y_time]
        input_args += [y_label[slot] for slot in slots]
        train_args = [lr] + input_args
        update_ratio = updater.get_update_ratio(params, model_updates)

        logging.info('Preparing %s train function.' % opt_type)
        t = time.time()
        self._train = theano.function(train_args, [cost_value, update_ratio],
                                      updates=model_updates)
        logging.info('Preparation done. Took: %.1f' % (time.time() - t))
        self._loss = theano.function(input_args, cost_value)

        logging.info('Preparing predict function.')
        t = time.time()
        predict_args = [x]
        predict_args += [x_score, x_switch]
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

    def prepare_data(self, seqs, slots):
        x = []
        x_score = []
        x_actor = []
        x_switch = []
        y_seq_id = []
        y_time = []
        y_labels = [[] for slot in slots]
        for item in seqs:
            x.append(item['data'])
            x_score.append(item['data_score'])
            x_actor.append(item['data_actor'])
            x_switch.append(item['data_switch'])
            for label in item['labels']:
                y_seq_id.append(len(x) - 1)
                y_time.append(label['time'])

                for i, slot in enumerate(slots):
                    y_labels[i].append(label['slots'][slot])

        x = padded(x, is_int=True).transpose(1, 0)

        x_score = padded(x_score).transpose(1, 0)
        x_actor = padded(x_actor, is_int=True).transpose(1, 0)
        x_switch = padded(x_switch, is_int=True).transpose(1, 0)

        return {
            'x': x,
            'x_score': np.array(x_score, dtype=np.float32)[:,:, np.newaxis],
            'x_actor': x_actor,
            'x_switch': np.array(x_switch, dtype=np.int32)[:,:, np.newaxis],
            'y_seq_id': y_seq_id,
            'y_time': y_time,
            'y_labels': y_labels,
        }




