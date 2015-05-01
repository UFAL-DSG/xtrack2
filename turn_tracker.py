import logging
import os
import time
from data import Data
import numpy as np
from passage.layers import *
from passage import updates
from passage.model import NeuralModel

import theano.tensor as tt


class Model(NeuralModel):
    def __init__(self, rnn_size, rnn_layers, vocab_size, lr):
        self.store_init_args(locals())
        x = T.imatrix()
        input_token_layer = Embedding(name="emb",
                                      size=rnn_size,
                                      n_features=vocab_size,
                                      input=x)

        lstms = []
        lstm_init_states = []
        prev_layer = input_token_layer
        for i in range(rnn_layers):
            lstm = LstmRecurrent(name='lstm_%d' % i,
                          size=rnn_size,
                          seq_output=True,
                          peepholes=True,
                          learn_init_state=False
                          )
            init_c = tt.matrix()
            init_h = tt.matrix()
            lstm.connect(prev_layer, init_c, init_h)
            lstms.append(lstm)
            prev_layer = lstm
            lstm_init_states.extend([init_c, init_h])


        out = Dense(name='output', size=vocab_size, activation='softmax')
        out.connect(prev_layer)

        y = T.imatrix()
        ux = UnBatch()
        ux.connect(out)

        uy = UnBatch(dtype='int32')
        uy.connect(IdentityInput(y, 1))

        loss = CrossEntropyObjective()
        loss.connect(ux, uy.output())
        loss_value = loss.output()

        lstm_states = []
        for lstm in lstms:
            lstm_states.extend(lstm.outputs)


        self.params = params = list(loss.get_params())

        self.curr_lr = theano.shared(lr)
        updater = updates.SGD(lr=self.curr_lr)

        model_updates = updater.get_updates(params, loss_value)
        update_ratio = updater.get_update_ratio(params, model_updates)
        self._train = theano.function(
            [x, y] + lstm_init_states,
            [loss_value, update_ratio] + lstm_states,
            updates=model_updates
        )

        self._predict = theano.function(
            [x] + lstm_init_states,
            [out.output()] + lstm_states
        )

    def set_lr(self, lr):
        self.curr_lr.set_value(lr)

    def get_lr(self):
        return self.curr_lr.get_value()

    def prepare_zero_states(self, data):
        n_states = data.shape[0]
        res = []
        for i in range(self.init_args['rnn_layers']):
            cells = np.zeros((n_states, self.init_args['rnn_size']), dtype='float32')
            hiddens = np.zeros((n_states, self.init_args['rnn_size']), dtype='float32')
            res.extend([cells, hiddens])

        return res



def main(experiment_path):
    theano.config.floatX = 'float32'

    train_path = os.path.join(experiment_path, 'train.json')
    xtd_t = Data.load(train_path)

    valid_path = os.path.join(experiment_path, 'dev.json')
    xtd_v = Data.load(valid_path)

    slots = xtd_t.slots
    classes = xtd_t.classes
    class_groups = xtd_t.slot_groups
    n_input_tokens = len(xtd_t.vocab)
    n_input_score_bins = len(xtd_t.score_bins)





if __name__ == '__main__':
    import utils
    utils.init_logging('LM')
    utils.pdb_on_error()

    y_time = tt.ivector()
    y_seq_id = tt.ivector()
    x = tt.imatrix()

    ts = TurnSquasher(100)
    ts.connect(IdentityInput(x, 1), y_time, y_seq_id)
    res = ts.output(dropout_active=True)

    f = theano.function([x, y_time, y_seq_id], res)

    xx = [
        [0],
        [2],
        [3],
        [4],
    ]

    res = f(xx, [0, 1, 2, 3], [0, 0, 0, 0])

    import ipdb; ipdb.set_trace()







    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--valid')
    parser.add_argument('--final_params', default='lm_params.p')
    parser.add_argument('--seq_length', type=int, default=20)
    parser.add_argument('--mb_size', type=int, default=20)

    parser.add_argument('--rnn_size', type=int, default=200)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1.0)

    args = parser.parse_args()

    main(**vars(args))