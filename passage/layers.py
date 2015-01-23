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
    def connect(self):
        pass

    def output(self, dropout_active=False):
        raise NotImplementedError()


class Embedding(object):

    def __init__(self, size=128, n_features=256, init='uniform'):
        self.init = getattr(inits, init)
        self.size = size
        self.n_features = n_features
        self.input = T.imatrix()
        self.wv = self.init((self.n_features, self.size))
        self.params = [self.wv]

    def output(self, dropout_active=False):
        return self.wv[self.input]

    def get_params(self):
        return self.params



class LstmRecurrent(object):

    def __init__(self, size=256, activation='tanh', gate_activation='steeper_sigmoid', init='normal', truncate_gradient=-1, seq_output=False, p_drop=0.):
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.gate_activation = getattr(activations, gate_activation)
        self.init = getattr(inits, init)
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.p_drop = p_drop

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size

        self.w_i = self.init((self.n_in, self.size))
        self.w_f = self.init((self.n_in, self.size))
        self.w_o = self.init((self.n_in, self.size))
        self.w_c = self.init((self.n_in, self.size))

        self.b_i = shared0s((self.size))
        self.b_f = shared0s((self.size))
        self.b_o = shared0s((self.size))
        self.b_c = shared0s((self.size))

        self.u_i = self.init((self.size, self.size))
        self.u_f = self.init((self.size, self.size))
        self.u_o = self.init((self.size, self.size))
        self.u_c = self.init((self.size, self.size))

        self.params = [self.w_i, self.w_f, self.w_o, self.w_c, 
            self.u_i, self.u_f, self.u_o, self.u_c,  
            self.b_i, self.b_f, self.b_o, self.b_c]

    def step(self, xi_t, xf_t, xo_t, xc_t, h_tm1, c_tm1, u_i, u_f, u_o, u_c):
        i_t = self.gate_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.gate_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.gate_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def output(self, dropout_active=False):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
        x_i = T.dot(X, self.w_i) + self.b_i
        x_f = T.dot(X, self.w_f) + self.b_f
        x_o = T.dot(X, self.w_o) + self.b_o
        x_c = T.dot(X, self.w_c) + self.b_c
        [out, cells], _ = theano.scan(self.step,
            sequences=[x_i, x_f, x_o, x_c], 
            outputs_info=[T.alloc(0., X.shape[1], self.size), T.alloc(0., X.shape[1], self.size)], 
            non_sequences=[self.u_i, self.u_f, self.u_o, self.u_c],
            truncate_gradient=self.truncate_gradient
        )
        if self.seq_output:
            return out
        else:
            return out[-1]

    def get_params(self):
        return set(self.params + self.l_in.get_params())


class Dense(object):
    def __init__(self, size=256, activation='rectify', init='orthogonal', p_drop=0.):
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
        self.indice2 = indices2
        self.size = data.size

    def output(self, dropout_active=False):
        out = self.data_layer.output(dropout_active=dropout_active)
        return out[self.indices, self.indice2]

    def get_params(self):
        return set(self.data_layer.get_params())



class CrossEntropyObjective(object):
    def connect(self, y_hat_layer, y_true):
        self.y_hat_layer = y_hat_layer
        self.y_true = y_true

    def output(self, dropout_active=False):
        return costs.CategoricalCrossEntropy(self.y_true,
                                             self.y_hat_layer.output(
                                                 dropout_active=dropout_active))

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