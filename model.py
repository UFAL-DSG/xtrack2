import logging
import time

import theano
import theano.tensor as tt

from passage.layers import *
from passage import updates
from passage.model import NeuralModel

class Model(NeuralModel):
    def __init__(self, slots, slot_classes, emb_size, n_input_tokens, updater):

        y_seq_id = tt.ivector()
        y_time = tt.ivector()
        y_label = {}
        for slot in slots:
            y_label[slot] = tt.ivector(name='y_label_%s' % slot)

        input_layer = Embedding(size=emb_size, n_features=n_input_tokens)
        x = input_layer.input

        lstm_layer = LstmRecurrent(size=16, seq_output=True)
        lstm_layer.connect(input_layer)

        cpt = CherryPick()
        cpt.connect(lstm_layer, y_time, y_seq_id)

        costs = []
        predictions = []
        for slot in slots:
            n_classes = len(slot_classes[slot])
            slot_mlp = MLP([16, 32, n_classes], ['tanh', 'tanh', 'softmax'])
            slot_mlp.connect(cpt)
            predictions.append(slot_mlp.output())

            slot_objective = CrossEntropyObjective()
            slot_objective.connect(slot_mlp, y_label[slot])
            costs.append(slot_objective)
        cost = SumOut()
        cost.connect(*costs)
        params = list(cost.get_params())
        cost_value = cost.output()

        updater = updates.RProp()
        model_updates = updater.get_updates(params, cost_value)


        logging.info('Preparing train function.')
        t = time.time()
        self._train = theano.function(
            [x, y_seq_id, y_time]
            + [y_label[slot] for slot in slots],
            cost_value, updates=model_updates)

        logging.info('Preparation done. Took: %.1f' % (time.time() - t))

        logging.info('Preparing predict function.')
        t = time.time()
        self._predict = theano.function(
            [x, y_seq_id, y_time],
            predictions
        )
        logging.info('Done. Took: %.1f' % (time.time() - t))




