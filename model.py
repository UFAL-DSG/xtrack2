import logging
import time

import theano
import theano.tensor as tt

from passage import updates
from passage.iterators import padded
from passage.layers import *
from passage.model import NeuralModel

class Model(NeuralModel):
    def __init__(self, slots, slot_classes, emb_size, n_input_tokens,
                 n_cells, lstm_n_layers, opt_type,
                 oclf_n_hidden, oclf_n_layers, oclf_activation, lr, debug,
                 p_drop, init_emb_from, vocab):

        y_seq_id = tt.ivector()
        y_time = tt.ivector()
        y_label = {}
        for slot in slots:
            y_label[slot] = tt.ivector(name='y_label_%s' % slot)

        input_layer = Embedding(name="emb", size=emb_size,
                                n_features=n_input_tokens)
        if init_emb_from:
            input_layer.init_from(init_emb_from, vocab)
        self.input_emb = input_layer.wv

        lstm_layer = None
        prev_layer = input_layer
        for i in range(lstm_n_layers):
            lstm_layer = LstmRecurrent(name="lstm", size=n_cells, seq_output=True,
                                       out_cells=False, p_drop=p_drop)
            lstm_layer.connect(prev_layer)
            prev_layer = lstm_layer

        assert lstm_layer is not None
        cpt = CherryPick()
        cpt.connect(lstm_layer, y_time, y_seq_id)

        costs = []
        predictions = []
        for slot in slots:
            n_classes = len(slot_classes[slot])
            slot_mlp = MLP([oclf_n_hidden] * oclf_n_layers + [n_classes],
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
        params = list(cost.get_params())
        n_params = sum(p.get_value().size for p in params)
        logging.info('This model has %d parameters.' % n_params)

        cost_value = cost.output(dropout_active=True)

        if opt_type == "rprop":
            updater = updates.RProp(lr=lr)
            model_updates = updater.get_updates(params, cost_value)
        elif opt_type == "sgd":
            #reg = updates.Regularizer(maxnorm=5.0)
            updater = updates.SGD(lr=lr, clipnorm=5.0)
        elif opt_type == "rmsprop":
            #reg = updates.Regularizer(maxnorm=5.0)
            updater = updates.RMSprop(lr=lr)  #, regularizer=reg)
        elif opt_type == "adam":
            #reg = updates.Regularizer(maxnorm=5.0)
            updater = updates.Adam(lr=lr)  #, regularizer=reg)
        else:
            raise Exception("Unknonw opt.")

        model_updates = updater.get_updates(params, cost_value)

        x = input_layer.input
        if debug:
            self._input_layer = theano.function([x], input_layer.output())

        train_args = [x, y_seq_id, y_time] + [y_label[slot] for slot in slots]
        update_ratio = updater.get_update_ratio(model_updates)

        logging.info('Preparing %s train function.' % opt_type)
        t = time.time()
        self._train = theano.function(train_args, [cost_value, update_ratio],
                                      updates=model_updates)
        logging.info('Preparation done. Took: %.1f' % (time.time() - t))

        logging.info('Preparing predict function.')
        t = time.time()
        self._predict = theano.function(
            [x, y_seq_id, y_time],
            predictions
        )
        logging.info('Done. Took: %.1f' % (time.time() - t))

    def init_loaded(self):
        pass

    def init_word_embeddings(self, w):
        self.input_emb.set_value(w)

    def prepare_data(self, seqs, slots):
        x = []
        y_seq_id = []
        y_time = []
        y_labels = [[] for slot in slots]
        for item in seqs:
            x.append(item['data'])
            for label in item['labels']:
                y_seq_id.append(len(x) - 1)
                y_time.append(label['time'])

                for i, slot in enumerate(slots):
                    y_labels[i].append(label['slots'][slot])

        x = padded(x).transpose(1, 0)

        return {
            'x': x,
            'y_seq_id': y_seq_id,
            'y_time': y_time,
            'y_labels': y_labels,
        }




