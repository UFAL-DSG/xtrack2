import logging
import time
import numpy as np
from passage.layers import *
from passage import updates
from passage.model import NeuralModel

import theano
import theano.tensor as tt


def print_data(data, d):
    for x in data:
        print d[x]


def load_data(fname):
    with open(fname) as f_in:
        return f_in.read().replace('\n', '<eos> ').split()


def build_dict(data):
    res = {}
    for w in data:
        if not w in res:
            res[w] = len(res)

    return res


def prepare_data(data, mb_size, vocab):
    res = np.zeros((len(data), ), dtype='int32')
    for i, w in enumerate(data):
        res[i] = vocab[w]

    res2 = np.zeros((np.floor(len(data) * 1.0 / mb_size), mb_size, ), dtype='int32')
    for i in range(mb_size):
        start = np.round((i) * len(data) * 1.0 / mb_size)
        finish = start + res2.shape[0]
        res2[0:res2.shape[0], i] = res[start:finish]

    return res2.T


class Model(NeuralModel):
    def __init__(self, rnn_size, rnn_layers, vocab_size, lr):
        self.store_init_args(locals())
        x = T.imatrix()
        input_token_layer = Embedding(name="emb",
                                      size=rnn_size,
                                      n_features=vocab_size,
                                      init=inits.uniform,
                                      input=x.T)

        lstms = []
        lstm_init_states = []
        prev_layer = input_token_layer
        for i in range(rnn_layers):
            lstm = LstmRecurrent(name='lstm_%d' % i,
                                 init=inits.uniform,
                                 size=rnn_size,
                                 seq_output=True,
                                 peepholes=False,
                                 learn_init_state=False
                          )
            init_c = tt.matrix()
            init_h = tt.matrix()
            lstm.connect(prev_layer, init_c, init_h)
            lstms.append(lstm)
            prev_layer = lstm
            lstm_init_states.extend([init_c, init_h])


        out = Dense(name='output', size=vocab_size, activation='softmax', init=inits.uniform)
        out.connect(prev_layer)

        y = T.imatrix()
        ux = UnBatch()
        ux.connect(out)

        ux2 = UnBatch()
        ux2.connect(IdentityInput(x.T, 1))

        uy = UnBatch(dtype='int32')
        uy.connect(IdentityInput(y.T, 1))

        loss = CrossEntropyObjective()
        loss.connect(ux, uy.output())
        loss_value = loss.output()

        self.params = params = list(loss.get_params())

        logging.info("# of parameters: %d" % sum([np.prod(p.get_value().shape) for p in params]))



        self._gc_grad = theano.function(
            [x, y] + lstm_init_states,
            tt.grad(loss_value, wrt=params),
        )

        self._gc = theano.function(
            [x, y] + lstm_init_states,
            [loss_value]
        )

        x_init = [np.array(np.random.randn(1, 200), dtype='float32') for i in range(4)]
        x_in = [range(10)]
        x_out = [range(10)]

        def check(p):
            dw_true = self._gc_grad(x_in, x_out, *x_init)[p]

            xd = 0.0001

            myw = params[p].get_value()
            sel_dims = []
            for i in range(myw.ndim):
                sel_dims.append(np.random.randint(myw.shape[i]))
            sel_dims = tuple(sel_dims)

            myw[sel_dims] += xd
            params[p].set_value(myw)
            loss_res = self._gc(x_in, x_out, *x_init)[0]

            myw[sel_dims] -= xd * 2
            params[p].set_value(myw)
            loss_res2 = self._gc(x_in, x_out, *x_init)[0]
            myw[sel_dims] += xd
            params[p].set_value(myw)

            dloss = (loss_res - loss_res2) / (2 * xd)

            print sel_dims, (dloss - dw_true[sel_dims]), dloss, dw_true[sel_dims]

        def gradchecks():
            for i in range(9):
                if len(params[i].get_value().shape) == 2:
                    print params[i].name
                    for ii in range(10):
                        check(i)
                    print '--'

        self.gradchecks = gradchecks

        lstm_final_states = []
        for lstm in lstms:
            for state in lstm.outputs:
                lstm_final_states.append(state[-1, :, :])




        self.curr_lr = theano.shared(lr)
        updater = updates.SGD(lr=self.curr_lr)

        model_updates = updater.get_updates(params, loss_value)
        update_ratio = updater.get_update_ratio(params, model_updates)
        grad_norm = updater.get_grad_norm()
        grad_vector = updater.get_grad_vector()
        self._train = theano.function(
            [x, y] + lstm_init_states,
            [loss_value, grad_norm, grad_vector, update_ratio] + lstm_final_states,
            updates=model_updates
        )



        pred_out = out.output()
        lstm_final_states = []
        for lstm in lstms:
            for state in lstm.outputs:
                lstm_final_states.append(state[-1, :, :])

        self._predict = theano.function(
            [x] + lstm_init_states,
            [pred_out] + lstm_final_states
        )

        #self._predictx = theano.function(
        #    [x, y] + lstm_init_states,
        #    [out.output(), ux2.output(), uy.output()] + lstm_final_states
        #)

    def set_lr(self, lr):
        self.curr_lr.set_value(lr)

    def get_lr(self):
        return self.curr_lr.get_value()

    def prepare_zero_states(self, data):
        n_seqs = data.shape[0]
        res = []
        for i in range(self.init_args['rnn_layers']):
            cells = np.zeros((n_seqs, self.init_args['rnn_size']), dtype='float32')
            hiddens = np.zeros((n_seqs, self.init_args['rnn_size']), dtype='float32')
            res.extend([cells, hiddens])

        return res

    def measure_perplexity(self, data, seq_length=20):
        res = 0.0
        cnt = 0.0
        pos = 0
        init_states = self.prepare_zero_states(data)
        while pos < data.shape[1]:
            x = data[:, pos:pos + seq_length]
            y = data[:, pos + 1:pos + seq_length + 1]

            if y.shape[1] == 0:
                assert pos == data.shape[1] - 1
                break

            if x.shape != y.shape:
                x = x[:, :y.shape[1]]

            assert x.shape == y.shape

            pred_res = self._predict(x, *init_states)
            preds = pred_res[0].swapaxes(0, 1)
            init_states = pred_res[1:]

            assert preds.shape[:-1] == y.shape #len(preds) == len(y)
            for bpred, by in zip(preds, y):
                for pred, y in zip(bpred, by):
                    res += np.log(pred[y])
                    cnt += 1

            pos += seq_length

        return np.exp(- res / cnt)




def main(train, valid, final_params, seq_length, mb_size,
         rnn_size, rnn_layers, lr):

    theano.config.floatX = 'float32'

    data_train = load_data(train)
    data_valid = load_data(valid)

    d = build_dict(data_train)
    d_rev = {val: key for key, val in d.iteritems()}
    logging.info("Size of vocabulary: %d" % len(d))

    data_train_x = prepare_data(data_train, mb_size, d)
    data_valid_x = prepare_data(data_valid, mb_size, d)


    logging.info('Size of training data: %10d' % data_train_x.shape[1])
    logging.info('Size of valid data:    %10d' % data_valid_x.shape[1])

    logging.info('Building model.')
    t = time.time()
    model = Model(rnn_size, rnn_layers, len(d), lr)
    logging.info('Building took: %.2f s' % (time.time() - t))

    epoch = 0
    while True:
        epoch += 1
        pos = 0
        model.save_params(final_params)
        logging.info('Measuring perplexity.')
        logging.info('Valid perplexity: %.5f' % model.measure_perplexity(data_valid_x, seq_length))
        init_states = model.prepare_zero_states(data_train_x)
        logging.info("Starting epoch %d" % epoch)
        while pos < data_train_x.shape[1]:
            x = data_train_x[:, pos:pos + seq_length]
            y = data_train_x[:, pos + 1:pos + seq_length + 1]
            if x.shape != y.shape:
                x = x[:, :y.shape[1]]
                #init_states = [i[:y.shape[1], :] for i in init_states]
            train_res = model._train(x, y, *init_states)
            loss, dw_norm, dw, ur = train_res[:4]
            init_states = train_res[4:]
            #init_states = [x[-1, :, :] for x in train_res[2:]]
            logging.info('epoch(%2d) pos(%5d) loss(%.4f) ratio(%.5f) dw(%.5f) %d%%' %
                        (epoch, pos, loss, ur, dw_norm, pos * 100.0 / data_train_x.shape[1])
            )

            pos += seq_length

        #model.gradchecks()
        #if epoch > 6:
        #    model.set_lr(model.get_lr() / 1.2)

    import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    import utils
    utils.init_logging('LM')
    utils.pdb_on_error()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--valid')
    parser.add_argument('--final_params', default='lm_params.p')
    parser.add_argument('--seq_length', type=int, default=20)
    parser.add_argument('--mb_size', type=int, default=15)

    parser.add_argument('--rnn_size', type=int, default=200)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1.0)

    args = parser.parse_args()

    main(**vars(args))

# x=model._predict(data_valid_x, *model.prepare_zero_states(data_valid_x))
