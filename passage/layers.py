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
        X = X / retain_prob * srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X

def theano_one_hot(idx, n):
    z = T.zeros((idx.shape[0], n))
    one_hot = T.set_subtensor(z[T.arange(idx.shape[0]), idx], 1)
    return one_hot

srng = RandomStreams()

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



class Embedding(object):

    def __init__(self, size=128, n_features=256, init='normal'):
        self.init = getattr(inits, init)
        self.size = size
        self.n_features = n_features
        self.input = T.imatrix()
        self.wv = self.init((self.n_features, self.size))
        self.params = {self.wv}

    def output(self, dropout_active=False):
        return self.wv[self.input]

    def get_params(self):
        return self.params



class LstmRecurrent(Layer):

    def __init__(self, size=256, init='normal', truncate_gradient=-1, seq_output=False, p_drop=0., init_scale=0.1, out_cells=False):
        self.init = getattr(inits, init)
        self.init_scale = init_scale
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.out_cells = out_cells
        self.p_drop = p_drop

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size

        self.w = self.init((self.n_in, self.size * 4), scale=self.init_scale,
                           name=self._name_param("W"))
        self.b = self.init((self.size * 4, ), scale=self.init_scale,
                           name=self._name_param("b"))

        #self.b_f = self.init((self.size, ), scale=self.init_scale)
        #self.b_i = self.init((self.size, ), scale=self.init_scale)
        #self.b_o = self.init((self.size, ), scale=self.init_scale)
        #self.b_m = self.init((self.size, ), scale=self.init_scale)

        self.u = self.init((self.size, self.size * 4), scale=self.init_scale,
                           name=self._name_param("U"))

        # Peep-hole connections.
        self.p = self.init((self.size, self.size * 3), scale=self.init_scale,
                           name=self._name_param("P"))

        self.params = [self.w, self.u, self.p, self.b]

    def _slice(self, x, n):
            return x[:, n * self.size:(n + 1) * self.size]

    def step(self, x_t, h_tm1, c_tm1, u, p):
        h_tm1_dot_u = T.dot(h_tm1, u)
        gates_fiom = x_t + h_tm1_dot_u

        g_f = self._slice(gates_fiom, 0)
        g_i = self._slice(gates_fiom, 1)
        g_o = self._slice(gates_fiom, 2)
        g_m = self._slice(gates_fiom, 3)

        c_tm1_dot_p = T.dot(c_tm1, p)

        g_f += self._slice(c_tm1_dot_p, 0)
        g_i += self._slice(c_tm1_dot_p, 1)
        g_o += self._slice(c_tm1_dot_p, 2)

        g_f = T.nnet.sigmoid(g_f)
        g_i = T.nnet.sigmoid(g_i)
        g_o = T.nnet.sigmoid(g_o)
        g_m = T.tanh(g_m)

        c_t = g_f * c_tm1 + g_i * g_m
        h_t = g_o * T.tanh(c_t)
        return h_t, c_t

    def output(self, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)

        x_dot_w = T.dot(X, self.w) + self.b
        [out, cells], _ = theano.scan(self.step,
            sequences=[x_dot_w],
            outputs_info=[T.alloc(0., X.shape[1], self.size), T.alloc(0., X.shape[1], self.size)], 
            non_sequences=[self.u, self.p],
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


class Dense(object):
    def __init__(self, size=256, activation='rectify', init='normal',
                 p_drop=0.):
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.init = getattr(inits, init)
        self.size = size
        self.p_drop = p_drop

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size
        if 'maxout' in self.activation_str:
            self.w = self.init((self.n_in, self.size*2))
            self.b = shared0s((self.size*2))
        else:
            self.w = self.init((self.n_in, self.size))
            self.b = shared0s((self.size))
        self.params = [self.w, self.b]

    def output(self, pre_act=False, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)

        is_tensor3_softmax = X.ndim > 2 and self.activation_str == 'softmax'

        shape = X.shape
        if is_tensor3_softmax: #reshape for tensor3 softmax
            X = X.reshape((shape[0]*shape[1], self.n_in))

        out =  self.activation(T.dot(X, self.w) + self.b)

        if is_tensor3_softmax: #reshape for tensor3 softmax
            out = out.reshape((shape[0], shape[1], self.size))

        return out

    def get_params(self):
        return set(self.params).union(set(self.l_in.get_params()))


class MLP(object):
    def __init__(self, sizes, activations):
        layers = []
        for size, activation in zip(sizes, activations):
            layer = Dense(size=size, activation=activation)
            layers.append(layer)

        self.stack = Stack(layers)

    def connect(self, l_in):
        self.stack.connect(l_in)

    def output(self, dropout_active=False):
        return self.stack.output(dropout_active=dropout_active)

    def get_params(self):
        return set(self.stack.get_params())


class Stack(object):
    def __init__(self, layers):
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


class CherryPick(object):
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



class CrossEntropyObjective(object):
    def connect(self, y_hat_layer, y_true):
        self.y_hat_layer = y_hat_layer
        self.y_true = y_true

    def output(self, dropout_active=False):
        y_hat_out = self.y_hat_layer.output(dropout_active=dropout_active)

        return costs.CategoricalCrossEntropy(self.y_true,
                                             y_hat_out)

    def get_params(self):
        return set(self.y_hat_layer.get_params())


class SumOut(object):
    def connect(self, *inputs):
        self.inputs = inputs

    def output(self, dropout_active=False):
        res = 0
        for l_in in self.inputs:
            res += l_in.output(dropout_active)

        return res

    def get_params(self):
        return set(flatten([layer.get_params() for layer in self.inputs]))