import theano
import theano.tensor as T
from theano.tensor.extra_ops import repeat
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import shared0s, flatten
import activations
import inits
import costs

import numpy as np

def dropout(X, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X * srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X

def theano_one_hot(idx, n):
    z = T.zeros((idx.shape[0], n))
    one_hot = T.set_subtensor(z[T.arange(idx.shape[0]), idx], 1)
    return one_hot

srng = RandomStreams(seed=1234)

class Layer(object):
    name = "unnamed_layer"
    #def connect(self):
    #    pass

    def output(self, dropout_active=False):
        raise NotImplementedError()

    def _name_param(self, param_name):
        return "%s__%s" % (self.name, param_name, )



class MatrixInput(object):
    def __init__(self, matrix):
        self.matrix = matrix
        self.size = matrix.shape[-1]

    def output(self, dropout_active=False):
        return T.as_tensor(self.matrix)

    def get_params(self):
        return set()


class IdentityInput(object):
    def __init__(self, val, size):
        self.val = val
        self.size = size

    def output(self, dropout_active=False):
        return self.val

    def get_params(self):
        return set()


class ZipLayer(object):
    def __init__(self, concat_axis, layers):
        self.layers = layers
        self.concat_axis = concat_axis
        self.size = sum(layer.size for layer in layers) # - layers[1].size

    def output(self, dropout_active=False):
        outs = [layer.output(dropout_active=dropout_active)
                for layer in self.layers]

        res = T.concatenate(outs, axis=self.concat_axis)
        return T.cast(res, dtype=theano.config.floatX)
        #return T.concatenate([outs[0] * T.repeat(outs[1], self.layers[0].size,
        #                                         axis=2),
        #                      outs[2]],
        #                     axis=self.concat_axis)

    def get_params(self):
        return set(flatten([layer.get_params() for layer in self.layers]))


class Embedding(Layer):

    def __init__(self, name=None, size=128, n_features=256, init='normal',
                 static=False, input=None):
        if name:
            self.name = name
        self.init = getattr(inits, init)
        self.size = size
        self.n_features = n_features
        self.input = input
        self.wv = self.init((self.n_features, self.size),
                            layer_width=self.size,
                            scale=1.0,
                            name=self._name_param("emb"))
        if static:
            self.params = set()
        else:
            self.params = {self.wv}
        self.static = static

    def output(self, dropout_active=False):
        return self.wv[self.input]

    def get_params(self):
        return self.params

    def init_from_dict(self, emb_dict):
        emb = self.wv.get_value()
        for k, v in emb_dict.iteritems():
            i = int(k)
            emb[i,:] = v

        # Normalize.
        emb_mean = emb.mean(axis=0)
        emb_std = emb.std(axis=0)
        emb = (emb - emb_mean) / (emb_std + 1e-7) * 100

        assert not (emb == np.nan).any()

        self.wv.set_value(emb)

    def init_from(self, f_name, vocab):
        emb = self.wv.get_value()
        import gzip
        with gzip.GzipFile(f_name) as f_in:
            for ln in f_in:
                ln = ln.strip().split()
                word = ln[0]
                if word in vocab:
                    word_id = vocab[word]
                    emb[word_id,:] = map(float, ln[1:])

        self.wv.set_value(emb)


class Recurrent(Layer):
    def __init__(self, name=None, size=256, init='normal', truncate_gradient=-1,
                 seq_output=False, p_drop=0., init_scale=0.1):
        self.size = size
        self.init = getattr(inits, init)
        self.name = name
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.p_drop = p_drop
        self.init_scale = init_scale

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size

        self.w = self.init((self.n_in, self.size),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("W"))

        self.u = self.init((self.size, self.size),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("U"))

        self.b = self.init((self.size, ),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("b"))

        self.params = [self.w, self.u]

    def output(self, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
            dropout_corr = 1.0
        else:
            dropout_corr = 1.0 - self.p_drop

        x_dot_w = T.dot(X, self.w * dropout_corr) + self.b
        out, _ = theano.scan(self.step,
            sequences=[x_dot_w],
            outputs_info=[T.alloc(0., X.shape[1], self.size)],
            non_sequences=[self.u],
            truncate_gradient=self.truncate_gradient
        )
        if self.seq_output:
            return out
        else:
            return out[-1]

    def step(self, x_t, h_tm1, u):
        h_t = activations.rectify(x_t + T.dot(h_tm1, u))
        return h_t

    def get_params(self):
        return self.l_in.get_params().union(self.params)



class LstmRecurrent(Layer):

    def __init__(self, name=None, size=256, init='normal', truncate_gradient=-1,
                 seq_output=False, p_drop=0., init_scale=0.1, out_cells=False,
                 peepholes=False):
        if name:
            self.name = name
        self.init = getattr(inits, init)
        self.init_scale = init_scale
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.out_cells = out_cells
        self.p_drop = p_drop
        self.peepholes = peepholes

        self.gate_act = activations.sigmoid
        self.modul_act = activations.tanh

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size

        # Input connections.
        self.w = self.init((self.n_in, self.size * 4),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("W"))

        self.b = self.init((self.size * 4, ),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("b"))

        # Initialize forget gates to large values.
        b = self.b.get_value()
        b[:self.size] = np.random.uniform(low=40.0, high=50.0, size=self.size)
        #b[self.size:] = 0.0
        self.b.set_value(b)

        # Recurrent connections.
        self.u = self.init((self.size, self.size * 4),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("U"))

        # Peep-hole connections.
        self.p_vec_f = self.init((self.size, ),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("peep_f"))
        self.p_vec_i = self.init((self.size, ),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("peep_i"))
        self.p_vec_o = self.init((self.size, ),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("peep_o"))

        self.init_c = self.init((self.size, ),
                                layer_width=self.size,
                                name=self._name_param("init_c"))
        self.init_h = self.init((self.size, ),
                                layer_width=self.size,
                                name=self._name_param("init_h"))

        #self.p = T.stack(T.diag(self.p_vec_f), T.diag(self.p_vec_i), T.diag(
        #    self.p_vec_o))


        self.params = [self.w, self.u, self.b]
        self.params += [self.init_c, self.init_h]
        if self.peepholes:
            self.params += [self.p_vec_f, self.p_vec_i, self.p_vec_o]

    def _slice(self, x, n):
            return x[:, n * self.size:(n + 1) * self.size]

    def step(self, x_t, h_tm1, c_tm1, u, p_vec_f, p_vec_i, p_vec_o):
        h_tm1_dot_u = T.dot(h_tm1, u)
        gates_fiom = x_t + h_tm1_dot_u

        g_f = self._slice(gates_fiom, 0)
        g_i = self._slice(gates_fiom, 1)
        g_m = self._slice(gates_fiom, 3)

        if self.peepholes:
            g_f += c_tm1 * p_vec_f
            g_i += c_tm1 * p_vec_i

        g_f = self.gate_act(g_f)
        g_i = self.gate_act(g_i)
        g_m = self.modul_act(g_m)

        c_t = g_f * c_tm1 + g_i * g_m

        g_o = self._slice(gates_fiom, 2)

        if self.peepholes:
            g_o += c_t * p_vec_o

        g_o = self.gate_act(g_o)

        h_t = g_o * T.tanh(c_t)
        return h_t, c_t

    def output(self, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
            dropout_corr = 1.0
        else:
            dropout_corr = 1.0 - self.p_drop

        x_dot_w = T.dot(X, self.w * dropout_corr) + self.b
        [out, cells], _ = theano.scan(self.step,
            sequences=[x_dot_w],
            #outputs_info=[
            #    T.alloc(0., X.shape[1], self.size),
            #    T.alloc(0.,X.shape[1], self.size)
            #],
            outputs_info=[
                T.repeat(self.init_c.dimshuffle('x', 0), X.shape[1], axis=0),
                T.repeat(self.init_h.dimshuffle('x', 0), X.shape[1], axis=0),
            ],
            non_sequences=[self.u, self.p_vec_f, self.p_vec_i, self.p_vec_o],
            truncate_gradient=self.truncate_gradient
        )
        if self.seq_output:
            if self.out_cells:
                return cells
            else:
                return out
        else:
            if self.out_cells:
                return cells[-1]
            else:
                return out[-1]

    def get_params(self):
        return self.l_in.get_params().union(self.params)


class BayLstmRecurrent(LstmRecurrent):
    def connect(self, l_in):
        super(BayLstmRecurrent, self).connect(l_in)

        self.p_w = inits.sharedX(float(np.random.randn(1)))
        self.p_b = inits.sharedX(float(np.random.randn(1)))

        self.params += [self.p_w, self.p_b]

    def step(self,
             x_t, x_prob_t, x_switch_t,
             h_tm1, c_tm1, ch_tm1,
             u, p_vec_f, p_vec_i, p_vec_o):

        h_t, c_t = super(BayLstmRecurrent, self).step(x_t, h_tm1, c_tm1, u,
                                                  p_vec_f,
                                           p_vec_i, p_vec_o)

        #x_switch_t = T.printing.Print('prob')(x_switch_t)
        x_prob_t = x_prob_t * self.p_w + self.p_b
        x_prob_t = x_prob_t.dimshuffle(0, 'x')
        c_t = T.switch(
            T.eq(x_switch_t, 1.0).dimshuffle(0, 'x'),
            x_prob_t * c_t + (1.0 - x_prob_t) * ch_tm1,
            c_t)
        ch_t = T.switch(
            T.eq(x_switch_t, 1.0).dimshuffle(0, 'x'),
            c_t,
            ch_tm1)

        return h_t, c_t, ch_t

    def output(self, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        # X: (time, seq, emb)
        #X = T.printing.Print('X')(X)

        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
            dropout_corr = 1.0
        else:
            dropout_corr = 1.0 - self.p_drop

        Xprob = X[:, :, -1]
        Xswitch = X[:, :, -2]
        X = T.set_subtensor(X[:, :, -1], 0)
        X = T.set_subtensor(X[:, :, -2], 0)

        x_dot_w = T.dot(X, self.w * dropout_corr) + self.b
        [out, cells, _], _ = theano.scan(self.step,
            sequences=[x_dot_w, Xprob, Xswitch],
            outputs_info=[
                T.alloc(0., X.shape[1], self.size),
                T.alloc(0., X.shape[1], self.size),
                T.alloc(0., X.shape[1], self.size)
            ],
            non_sequences=[self.u, self.p_vec_f, self.p_vec_i, self.p_vec_o],
            truncate_gradient=self.truncate_gradient
        )

        if self.seq_output:
            if self.out_cells:
                return cells
            else:
                return out
        else:
            if self.out_cells:
                return cells[-1]
            else:
                return out[-1]



class Dense(Layer):
    def __init__(self, name=None, size=256, activation='rectify', init='normal',
                 p_drop=0.):
        if name:
            self.name = name
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.init = getattr(inits, init)
        self.size = size
        self.p_drop = p_drop

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size

        self.w = self.init(
            (self.n_in, self.size),
            layer_width=self.size,
            name=self._name_param("w")
        )
        self.b = self.init(
            (self.size, ),
            layer_width=self.size,
            name=self._name_param("b")
        )
        self.params = [self.w, self.b]

    def output(self, pre_act=False, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
            dropout_corr = 1.
        else:
            dropout_corr = 1.0 - self.p_drop

        is_tensor3_softmax = X.ndim > 2 and self.activation_str == 'softmax'

        shape = X.shape
        if is_tensor3_softmax: #reshape for tensor3 softmax
            X = X.reshape((shape[0]*shape[1], self.n_in))

        out =  self.activation(T.dot(X, self.w * dropout_corr) + self.b)

        if is_tensor3_softmax: #reshape for tensor3 softmax
            out = out.reshape((shape[0], shape[1], self.size))

        return out

    def get_params(self):
        return set(self.params).union(set(self.l_in.get_params()))


class MLP(Layer):
    def __init__(self, sizes, activations, name=None, p_drop=0.):
        layers = []
        for layer_id, (size, activation) in enumerate(zip(sizes, activations)):
            layer = Dense(size=size, activation=activation, name="%s_%d" % (
                name, layer_id, ), p_drop=p_drop)
            layers.append(layer)

        self.stack = Stack(layers, name=name)
        self.size = layers[-1].size

    def connect(self, l_in):
        self.stack.connect(l_in)

    def output(self, dropout_active=False):
        return self.stack.output(dropout_active=dropout_active)

    def get_params(self):
        return set(self.stack.get_params())


class Stack(Layer):
    def __init__(self, layers, name=None):
        if name:
            self.name = name
        self.layers = layers
        self.size = layers[-1].size


    def connect(self, l_in):
        self.layers[0].connect(l_in)
        for i in range(1, len(self.layers)):
            self.layers[i].connect(self.layers[i-1])

    def output(self, dropout_active=False):
        return self.layers[-1].output(dropout_active=dropout_active)

    def get_params(self):
        return set(flatten([layer.get_params() for layer in self.layers]))


class CherryPick(Layer):
    def connect(self, data, indices, indices2):
        self.data_layer = data
        self.indices = indices
        self.indices2 = indices2
        self.size = data.size

    def output(self, dropout_active=False):
        out = self.data_layer.output(dropout_active=dropout_active)
        return out[self.indices, self.indices2]

    def get_params(self):
        return set(self.data_layer.get_params())



class CrossEntropyObjective(Layer):
    def connect(self, y_hat_layer, y_true):
        self.y_hat_layer = y_hat_layer
        self.y_true = y_true

    def output(self, dropout_active=False):
        y_hat_out = self.y_hat_layer.output(dropout_active=dropout_active)

        return costs.CategoricalCrossEntropy(self.y_true,
                                             y_hat_out)

    def get_params(self):
        return set(self.y_hat_layer.get_params())


class SumOut(Layer):
    def connect(self, *inputs, **kwargs):
        self.inputs = inputs
        self.scale = kwargs.get('scale', 1.0)

    def output(self, dropout_active=False):
        res = 0
        for l_in in self.inputs:
            res += l_in.output(dropout_active)

        return res * self.scale

    def get_params(self):
        return set(flatten([layer.get_params() for layer in self.inputs]))