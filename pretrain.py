import inspect
import logging
import os
import random
import time

import theano
import theano.tensor as tt

from passage import updates
from passage.iterators import padded
from passage.layers import *
from passage.lstm_with_confidence import LstmWithConfidence
from passage.model import NeuralModel
from passage.utils import intX, floatX, iter_data

from data import Data

from scipy.spatial.distance import cosine



class Model(NeuralModel):
    def __init__(self, emb_size,
                 input_n_layers, input_n_hidden, input_activation,
                 lstm_n_cells, lstm_n_layers,
                 oclf_n_hidden, oclf_n_layers, oclf_activation,
                 vocab):
        self.store_init_args(locals())

        self.vocab = vocab
        n_input_tokens = len(vocab)

        x = tt.imatrix(name='x')  # time x dialog
        input_args = []
        input_zip_layers = []
        input_args.append(x)
        input_token_layer = Embedding(name="emb",
                                      size=emb_size,
                                      n_features=n_input_tokens,
                                      input=x)
        input_zip_layers.append(input_token_layer)

        x_conf = tt.matrix(name='x_conf')
        input_args.append(x_conf)
        input_conf_layer = IdentityInput(x_conf[:, :, np.newaxis], 1)
        input_zip_layers.append(input_conf_layer)

        prev_layer = ZipLayer(2, input_zip_layers)

        flatten =  ReshapeLayer(x.shape[0] * x.shape[1], prev_layer.size)
        flatten.connect(prev_layer)
        prev_layer = flatten

        input_mlp_layer = MLP([input_n_hidden  ] * input_n_layers,
                              [input_activation] * input_n_layers,
                              [0.0             ] * input_n_layers,
                              name="input_mlp")
        input_mlp_layer.connect(prev_layer)
        prev_layer = input_mlp_layer

        reshape_back = ReshapeLayer(x.shape[0], x.shape[1], input_n_hidden)
        reshape_back.connect(prev_layer)
        prev_layer = reshape_back

        # Forward LSTM layer.
        assert lstm_n_layers > 0
        for i in range(lstm_n_layers):
            lstm_layer = LstmRecurrent(
                name="lstm%d" % i,
                size=lstm_n_cells,
                seq_output=True,
                out_cells=False)
            lstm_layer.connect(prev_layer)
            prev_layer = lstm_layer

        assert prev_layer is not None

        y = x[1:, :]

        mlp = MLP([oclf_n_hidden  ] * oclf_n_layers + [n_input_tokens],
                       [oclf_activation] * oclf_n_layers + ['softmax'],
                       [0.0            ] * oclf_n_layers + [0.0      ],
                       name="mlp")
        mlp.connect(prev_layer)

        predictions = mlp.output()

        mlp_flat_out = tt.reshape(predictions[:x.shape[0] - 1], ((x.shape[0] - 1) * x.shape[1], mlp.size))
        y_flat = tt.reshape(y, ((x.shape[0] - 1) * x.shape[1], ))

        cost_value = costs.CategoricalCrossEntropy(y_flat, mlp_flat_out)


        self.params = params = list(mlp.get_params())
        n_params = sum(p.get_value().size for p in params)
        logging.info('This model has %d parameters:' % n_params)
        for param in sorted(params, key=lambda x: x.name):
            logging.info('  - %20s: %10d' % (param.name, param.get_value(

            ).size, ))

        lr = tt.scalar('lr')
        clip = 5.0
        reg = updates.Regularizer()
        updater = updates.Adam(lr=lr, clip=clip, regularizer=reg)  #,

        loss_args = list(input_args)

        model_updates = updater.get_updates(params, cost_value)

        train_args = [lr] + loss_args
        update_ratio = updater.get_update_ratio(params, model_updates)

        t = time.time()
        self._train = theano.function(train_args, [cost_value / y.shape[0], update_ratio],
                                              updates=model_updates)
        logging.info('Preparation done. Took: %.1f' % (time.time() - t))

        self._loss = theano.function(loss_args, cost_value)

        logging.info('Preparing predict function.')
        t = time.time()
        predict_args = list(input_args)
        self._predict = theano.function(
            predict_args,
            predictions
        )
        logging.info('Done. Took: %.1f' % (time.time() - t))

    def prepare_data_train(self, seqs, debug_data=False, dense_labels=False):
        return self._prepare_data(seqs)

    def prepare_minibatches(self, data, mb_size):
        minibatches = []
        seqs_mb = iter_data(data.sequences, size=mb_size)
        for mb in seqs_mb:
            data = self.prepare_data_train(mb)
            minibatches.append(data)

        return minibatches

    def prepare_data_predict(self, seqs):
        return self._prepare_data(seqs)

    def _prepare_data(self, seqs):
        x = []
        x_score = []

        for item in seqs:
            data = []
            data_score = []
            assert len(item['data']) == len(item['data_score'])

            for words, score in zip(item['data'], item['data_score']): #, item['ftrs']):
                assert len(words) == 1
                x.append(words[0])
                x_score.append(score[0])

            x.append(0)
            x_score.append(0.0)

        n_seqs = len(seqs)
        n_words_per_seq = len(x) / n_seqs
        x = x[:n_seqs * n_words_per_seq]
        x_score = x_score[:n_seqs * n_words_per_seq]

        x = intX(x).reshape((len(seqs), n_words_per_seq, )).transpose(1, 0)
        x_score = floatX(x_score).reshape((len(seqs), n_words_per_seq, )).transpose(1, 0)

        data = [x, x_score]

        return tuple(data)


class Pretrain(object):
    def __init__(self, **kwargs):
        experiment_path = kwargs.pop('experiment_path')
        params_file = kwargs.pop('params_file')
        mb_size = kwargs.pop('mb_size')
        lr = kwargs.pop('lr')

        train_path = os.path.join(experiment_path, 'train.json')
        xtd_t = Data.load(train_path)

        valid_path = os.path.join(experiment_path, 'dev.json')
        xtd_v = Data.load(valid_path)

        model = Model(vocab=xtd_t.vocab, **kwargs)

        train_minibatches = model.prepare_minibatches(xtd_t, mb_size)
        #vocab_rev = {v: k for k, v in model.vocab.iteritems()}
        test_words = ['indian', 'chinese', 'food', 'restaurant', 'north', 'east', 'west']

        epoch = 0
        while True:
            epoch += 1
            params = model.dump_params()['model_params']


            for w1 in test_words:
                for w2 in test_words:
                    v1 = params['emb__emb'][model.vocab[w1]]
                    v2 = params['emb__emb'][model.vocab[w2]]
                    logging.debug('   Cosine(%s, %s) = %.2f' % (w1, w2, cosine(v1, v2)))

            logging.debug('Epoch %s, leraning rate: %.2f' % (epoch, lr, ))
            random.shuffle(train_minibatches)
            losses = []
            upds = []
            for mb in train_minibatches:
                loss, upd = model._train(lr, *mb)
                losses.append(loss)
                upds.append(upd)

            logging.info("  mean_loss(%.2f) mean_update_ratio(%.5f)" % (np.mean(losses), np.mean(upds), ))

            logging.info('Saving params: %s' % params_file)
            model.save_params(params_file)
            lr = lr * 0.95


def main(**kwargs):
    pretrain = Pretrain(**kwargs)


if __name__ == '__main__':
    import utils
    utils.init_logging("PreTrain")
    utils.pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path')
    parser.add_argument('params_file')
    parser.add_argument('--emb_size', default=170, type=int)
    parser.add_argument('--mb_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    parser.add_argument('--input_n_layers', default=2, type=int)
    parser.add_argument('--input_n_hidden', default=300, type=int)
    parser.add_argument('--input_activation', default='rectify')

    parser.add_argument('--lstm_n_cells', default=300, type=int)
    parser.add_argument('--lstm_n_layers', default=1, type=int)

    parser.add_argument('--oclf_n_hidden', default=100, type=int)
    parser.add_argument('--oclf_n_layers', default=0, type=int)
    parser.add_argument('--oclf_activation', default='rectify')

    args = parser.parse_args()

    main(**vars(args))