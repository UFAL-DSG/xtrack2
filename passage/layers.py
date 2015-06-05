import theano
import theano.tensor as T
from theano.tensor.extra_ops import repeat
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import itertools
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

    def set_val(self, val):
        self.val = val

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


class Broadcast(object):
    def __init__(self, n_times):
        self.n_times = n_times

    def connect(self, layer):
        self.layer = layer

    def output(self, dropout_active=False):
        out = self.layer.output(dropout_active=dropout_active)
        return T.concatenate([out] * self.n_times, axis=len(out.shape) - 1)

    def get_params(self):
        return self.layer.get_params()


class UnBatch(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def connect(self, layer_x):
        self.layer_x = layer_x
        self.size = layer_x.size

    def output(self, dropout_active=False):
        x = self.layer_x.output(dropout_active=dropout_active)
        new_shape = list(x.shape)
        new_shape[1] = new_shape[0] * new_shape[1]
        new_shape = tuple(new_shape[1:])
        res = x.reshape(new_shape)

        dtype = theano.config.floatX
        if self.dtype:
            dtype = self.dtype

        return T.cast(res, dtype=dtype)

    def get_params(self):
        return self.layer_x.get_params()


class SumLayer(object):
    def __init__(self, layers):
        self.layers = layers
        self.size = layers[0].size

    def output(self, dropout_active=False):
        outs = [layer.output(dropout_active=dropout_active)
                for layer in self.layers]

        res = outs[0]
        for out in outs[1:]:
            res += out

        return T.cast(res, dtype=theano.config.floatX)

    def get_params(self):
        return set(flatten([layer.get_params() for layer in self.layers]))



class OneHot(Layer):
    def __init__(self, name=None, n_features=256, input=None):
        if name:
            self.name = name
        self.input = input
        self.size = n_features

    def output(self, dropout_active=False):
        inp = T.flatten(self.input)

        res = T.zeros((self.input.shape[0] * self.input.shape[1], self.size, ))
        res = T.set_subtensor(res[T.arange(inp.shape[0]), inp], 1.0)

        res = T.reshape(res, (self.input.shape[0], self.input.shape[1], self.size, ))

        return res

    def get_params(self):
        return set()


class Embedding(Layer):
    def __init__(self, name=None, size=128, n_features=256, init=inits.normal,
                 static=False, input=None):
        if name:
            self.name = name
        self.init = init
        self.size = size
        self.n_features = n_features
        self.input = input
        self.wv = self.init((self.n_features, self.size),
                            fan_in=self.n_features,
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
    def __init__(self, name=None, size=256, init=inits.normal, truncate_gradient=-1,
                 seq_output=False, p_drop=0., init_scale=0.1):
        self.size = size
        self.init = init
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

    def __init__(self, name=None, size=256, init=inits.normal, truncate_gradient=-1,
                 seq_output=False, p_drop=0., init_scale=0.1, out_cells=False,
                 peepholes=False, enable_branch_exp=False, backward=False,
                 learn_init_state=True):
        if name:
            self.name = name
        self.init = init
        self.init_scale = init_scale
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.out_cells = out_cells
        self.p_drop = p_drop
        self.peepholes = peepholes
        self.backward = backward
        self.learn_init_state = learn_init_state

        self.gate_act = activations.sigmoid
        self.modul_act = activations.tanh

        self.enable_branch_exp = enable_branch_exp
        self.lagged = []

    def _init_input_connections(self, n_in):
        self.w = self.init((n_in, self.size * 4),
                           fan_in=n_in,
                           name=self._name_param("W"))
        self.b = inits.const((self.size * 4, ),
                             0.1,
                             name=self._name_param("b"))

        #self.br = self.init((self.size * 4, ),
        #                   layer_width=self.size,
        #                   scale=self.init_scale,
        #                   name=self._name_param("br"))

        # Initialize forget gates to large values.
        b = self.b.get_value()
        b[:self.size] = 1.0
        #b[self.size:] = 0.0
        self.b.set_value(b)

    def _init_recurrent_connections(self):
        self.u = self.init((self.size, self.size * 4),
                           fan_in=self.size,
                           name=self._name_param("U"))

    def _init_peephole_connections(self):
        self.p_vec_f = self.init((self.size, ),
                                 fan_in=self.size,
                                 name=self._name_param("peep_f"))
        self.p_vec_i = self.init((self.size, ),
                                 fan_in=self.size,
                                 name=self._name_param("peep_i"))
        self.p_vec_o = self.init((self.size, ),
                                 fan_in=self.size,
                                 name=self._name_param("peep_o"))

    def _init_initial_states(self, init_c=None, init_h=None):
        if self.learn_init_state:
            self.init_c = self.init((self.size, ),
                                    fan_in=self.size,
                                    name=self._name_param("init_c"))
            self.init_h = self.init((self.size, ),
                                    fan_in=self.size,
                                    name=self._name_param("init_h"))
        else:
            self.init_c = init_c
            self.init_h = init_h

    def connect(self, l_in, init_c=None, init_h=None):
        self.l_in = l_in

        self._init_input_connections(l_in.size)
        self._init_recurrent_connections()
        self._init_peephole_connections()  # TODO: Make also conditional.

        self.params = [self.w, self.u, self.b]

        self._init_initial_states(init_c, init_h)
        if self.learn_init_state:
            self.params += [self.init_c, self.init_h]

        if self.peepholes:
            self.params += [self.p_vec_f, self.p_vec_i, self.p_vec_o]

    def connect_lagged(self, l_in):
        self.lagged.append(l_in)

    def _slice(self, x, n):
            return x[:, n * self.size:(n + 1) * self.size]

    def step(self, x_t, h_tm1, c_tm1, u, p_vec_f, p_vec_i, p_vec_o,
             dropout_active):
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

    def _compute_x_dot_w(self, dropout_active):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
            dropout_corr = 1.0
        else:
            dropout_corr = 1.0 - self.p_drop
        x_dot_w = T.dot(X, self.w * dropout_corr) + self.b
        return x_dot_w

    def _reverse_if_backward(self, cells, out):
        if self.backward:
            out = out[::-1, ]
            cells = cells[::-1, ]
        return cells, out

    def _prepare_result(self, cells, out):
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

    def _prepare_outputs_info(self, x_dot_w):
        if self.learn_init_state:
            outputs_info = [
                T.repeat(self.init_c.dimshuffle('x', 0), x_dot_w.shape[1], axis=0),
                T.repeat(self.init_h.dimshuffle('x', 0), x_dot_w.shape[1], axis=0),
            ]
        else:
            outputs_info = [
                self.init_c,
                self.init_h
            ]
        return outputs_info

    def _process_scan_output(self, res):
        (out, cells), _ = res

        return out, cells

    def _compute_seq(self, x_dot_w, dropout_active):
        outputs_info = self._prepare_outputs_info(x_dot_w)

        res = theano.scan(self.step,
                                      sequences=[x_dot_w],
                                      outputs_info=outputs_info,
                                      non_sequences=[self.u, self.p_vec_f,
                                                     self.p_vec_i,
                                                     self.p_vec_o,
                                                     1 if dropout_active else
                                                     0],
                                      truncate_gradient=self.truncate_gradient,
                                      go_backwards=self.backward
        )
        out, cells = self._process_scan_output(res)
        return cells, out

    def output(self, dropout_active=False):
        x_dot_w = self._compute_x_dot_w(dropout_active)

        cells, out = self._compute_seq(x_dot_w, dropout_active)
        cells, out = self._reverse_if_backward(cells, out)

        self.outputs = [cells, out]

        return self._prepare_result(cells, out)

    def get_params(self):
        return self.l_in.get_params().union(self.params)



class Dense(Layer):
    def __init__(self, name=None, size=256, activation='rectify', init=inits.normal,
                 p_drop=0.):
        if name:
            self.name = name
        self.activation_str = activation
        self.activation = getattr(activations, activation)
        self.init = init
        self.size = size
        self.p_drop = p_drop

    def connect(self, l_in):
        self.l_in = l_in
        self.n_in = l_in.size

        self.w = self.init(
            (self.n_in, self.size),
            fan_in=self.n_in,
            name=self._name_param("w")
        )
        self.b = inits.const(
            (self.size, ),
            val=0.1,
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
    def __init__(self, sizes, activations, p_drop=itertools.repeat(0.0),
                 name=None, init=inits.normal):
        self.layers = layers = []
        for layer_id, (size, activation, l_p_drop) in enumerate(zip(sizes,
                                                           activations, p_drop)):
            layer = Dense(size=size, activation=activation, name="%s_%d" % (
                name, layer_id, ), p_drop=l_p_drop, init=init)
            layers.append(layer)

        self.stack = Stack(layers, name=name)

    def connect(self, l_in):
        self.l_in = l_in
        self.stack.connect(l_in)

        if len(self.layers) != 0:
            self.size = self.layers[-1].size
        else:
            self.size = self.l_in.size

    def output(self, dropout_active=False):
        return self.stack.output(dropout_active=dropout_active)

    def get_params(self):
        return set(self.stack.get_params())


class Stack(Layer):
    def __init__(self, layers, name=None):
        if name:
            self.name = name
        self.layers = layers

    def connect(self, l_in):
        self.l_in = l_in
        if len(self.layers) > 0:
            self.layers[0].connect(l_in)
            for i in range(1, len(self.layers)):
                self.layers[i].connect(self.layers[i-1])

            self.size = self.layers[-1].size
        else:
            self.size = l_in.size
            self.layers = [l_in]

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


class CherryPickDelta(Layer):
    def connect(self, data, indices, indices2):
        self.data_layer = data
        self.indices = indices
        self.indices2 = indices2
        self.size = data.size * 2

    def output(self, dropout_active=False):
        out = self.data_layer.output(dropout_active=dropout_active)
        res = out[self.indices, self.indices2]
        res_delta = T.concatenate([T.zeros_like(out[0, 0]).dimshuffle('x', 0), out[self.indices[1:], self.indices2[1:]]])

        return T.concatenate([res, res_delta], axis=1)

    def get_params(self):
        return set(self.data_layer.get_params())



class SVMObjective(Layer):
    def connect(self, y_hat_layer, y_true):
        self.y_hat_layer = y_hat_layer
        self.y_true = y_true

    def output(self, dropout_active=False):
        y_hat_out = self.y_hat_layer.output(dropout_active=dropout_active)
        f_yi = y_hat_out[T.arange(self.y_true.shape[0]), self.y_true]

        res = (y_hat_out - f_yi.dimshuffle(0, 'x') + 1)

        res = res * (res > 0)

        return res.sum()

    def get_params(self):
        return set(self.y_hat_layer.get_params())


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

class InvCrossEntropyObjective(Layer):
    def connect(self, y_hat_layer, y_true):
        self.y_hat_layer = y_hat_layer
        self.y_true = y_true

    def output(self, dropout_active=False):
        y_hat_out = self.y_hat_layer.output(dropout_active=dropout_active)

        return costs.CategoricalCrossEntropy(self.y_true,
                                             (1 - y_hat_out))

    def get_params(self):
        return set(self.y_hat_layer.get_params())


class WeightedCrossEntropyObjective(Layer):
    def connect(self, y_hat_layer, y_true, y_weights):
        self.y_hat_layer = y_hat_layer
        self.y_true = y_true
        self.y_weights = y_weights

    def output(self, dropout_active=False):
        y_hat_out = self.y_hat_layer.output(dropout_active=dropout_active)

        return costs.WeightedCategoricalCrossEntropy(
            self.y_true,
            y_hat_out,
            self.y_weights
        )

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


class SeqUnwrapper(Layer):
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def connect(self, prev_layer, y_time, y_seq_id):
        self.prev_layer = prev_layer
        self.y_time = y_time
        self.y_seq_id = y_seq_id

        self.size = prev_layer.size

    def output(self, dropout_active=False):
        out = self.prev_layer.output(dropout_active=dropout_active)

        res, _ = theano.scan(self._step,
                          sequences=[self.y_time, self.y_seq_id],
                          non_sequences=[out]
        )
        return res

    def _step(self, y_t, y_seq_id, x):
        res = x[0:y_t, y_seq_id,:]
        padding = T.zeros((self.seq_len - y_t, self.size))
        return T.concatenate([res, padding])

    def get_params(self):
        return self.prev_layer.get_params()


class SeqMaxPooling(Layer):
    def connect(self, prev_layer, x_score, y_time, y_seq_id):
        self.prev_layer = prev_layer
        self.x_score = x_score
        self.y_time = y_time
        self.y_seq_id = y_seq_id

        self.size = prev_layer.size

    def output(self, dropout_active=False):
        out = self.prev_layer.output(dropout_active=dropout_active)

        res, _ = theano.scan(self._step,
                          sequences=[self.y_time, self.y_seq_id],
                          non_sequences=[out, self.x_score]
        )
        return res

    def _step(self, y_t, y_seq_id, x, x_score):
        score = x_score[0:y_t, y_seq_id].dimshuffle(0, 'x')
        return T.max(x[0:y_t, y_seq_id,:] * (score * 0 + 1), axis=0)

    def get_params(self):
        return self.prev_layer.get_params()


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def connect(self, prev_layer, rng, filter_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """


        self.prev_layer = prev_layer

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)



        # store parameters of this layer
        self.params = [self.W, self.b]

        self.filter_shape = filter_shape
        self.poolsize = poolsize

        self.size = 600 #15 * filter_shape[0] #fan_out

    def get_params(self):
        return self.prev_layer.get_params().union(self.params)


    def output(self, dropout_active=False):
        input = self.prev_layer.output(dropout_active)
        input_shape = [input.shape[0], 1, input.shape[1], input.shape[2]]

        input = input.reshape(input_shape)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=self.filter_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        res = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        return res.flatten(2)


class ProbLayer(Layer):
    def __init__(self, name=None, size=128, init=inits.normal, input=None):
        self.name = name
        self.init = init
        self.size = size
        self.input = input

    def connect(self, prev_layer):
        self.prev_layer = prev_layer

        #self.w = self.init((prev_layer.size, self.size),
        #                    layer_width=self.size,
        #                    scale=1.0,
        #                    name=self._name_param("W"))
        #self.b = self.init((self.size, ),
        #                   layer_width=self.size,
        #                   name=self._name_param("b"))

        self.wc = self.init((1, prev_layer.size, ),
                            layer_width=prev_layer.size,
                            scale=1.0,
                            name=self._name_param("Wc"))
        #self.wc.set_value(np.ones_like(self.wc.get_value(), dtype='float32'))

        self.bc = self.init((prev_layer.size, ),
                             layer_width=1,
                             scale=1.0,
                             name=self._name_param("bc"))
        #self.bc.set_value(np.zeros_like(self.bc.get_value(), dtype='float32'))
        #self.wc = theano.shared(np.array(1.0, dtype='float32'), name=self._name_param("Wc"))
        #self.bc = theano.shared(np.array(0.0, dtype='float32'), name=self._name_param("bc"))



        self.params = [self.wc, self.bc]

    def output(self, dropout_active=False):
        x = self.prev_layer.output(dropout_active=dropout_active)

        #y = activations.sigmoid(T.dot(x, self.w) + self.b)

        cy = activations.sigmoid(T.dot(self.input.dimshuffle(0, 1, 2, 'x'), self.wc) + self.bc)

        return x * cy

    def get_params(self):
        return self.prev_layer.get_params().union(self.params)


class MaxPooling(Layer):
    def __init__(self, name, pool_dimension):
        self.name = name
        self.pool_dimension = pool_dimension
        self.dummy_input = False

    def set_dummy_input(self, x):
        self.dummy_input = x

    def connect(self, prev_layer):
        self.prev_layer = prev_layer
        self.size = prev_layer.size

    def output(self, dropout_active=False):
        if not self.dummy_input:
            x = self.prev_layer.output(dropout_active=dropout_active)
        else:
            x = self.dummy_input

        self.input_var = x
        return T.max(x, axis=self.pool_dimension)

    def get_params(self):
        return self.prev_layer.get_params()



class NGramLSTM(Layer):

    def __init__(self, name=None, size=256, init=inits.normal, truncate_gradient=-1,
                 seq_output=False, p_drop=0., init_scale=0.1, out_cells=False,
                 peepholes=False, enable_branch_exp=False, backward=False):
        if name:
            self.name = name
        self.init = init
        self.init_scale = init_scale
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.out_cells = out_cells
        self.p_drop = p_drop
        self.peepholes = peepholes
        self.backward = backward

        self.gate_act = activations.sigmoid
        self.modul_act = activations.sigmoid #tanh

        self.enable_branch_exp = enable_branch_exp
        self.lagged = []

    def connect(self, l_in):
        self.l_in = l_in

        self._init_input_connections(l_in.size)
        self._init_recurrent_connections()
        self._init_peephole_connections()  # TODO: Make also conditional.
        self._init_initial_states()

        self.filters = self.init((l_in.size * 3, self.size * 4),
                                 layer_width=self.size * 4,
                                 name=self._name_param("filters"))

        #self.params = [self.w, self.u, self.b]
        self.params = [self.u]
        self.params += [self.init_c, self.init_h]
        self.params += [self.filters]
        self.params += [self.b]

        if self.peepholes:
            self.params += [self.p_vec_f, self.p_vec_i, self.p_vec_o]

    def _init_input_connections(self, n_in):
        self.w = self.init((n_in, self.size * 4),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("W"))
        self.b = self.init((self.size * 4, ),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("b"))

        # Initialize forget gates to large values.
        #b = self.b.get_value()
        #b[:self.size] = np.random.uniform(low=40.0, high=50.0, size=self.size)
        #b[self.size:] = 0.0
        #self.b.set_value(b)

    def _init_recurrent_connections(self):
        self.u = self.init((self.size, self.size * 4),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("U"))

    def _init_peephole_connections(self):
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

    def _init_initial_states(self):
        self.init_c = self.init((self.size, ),
                                layer_width=self.size,
                                name=self._name_param("init_c"))
        self.init_h = self.init((self.size, ),
                                layer_width=self.size,
                                name=self._name_param("init_h"))



    def connect_lagged(self, l_in):
        self.lagged.append(l_in)

    def _slice(self, x, n):
            return x[:, n * self.size:(n + 1) * self.size]

    def step(self, x_tm2, x_tm1, x_tm0, h_tm1, c_tm1, u, p_vec_f, p_vec_i, p_vec_o,
             dropout_active):
        #x_t = T.max(T.concatenate([[x_tm2], [x_tm1], [x_tm0]]), axis=0)
        x_t = T.dot(T.concatenate([x_tm2, x_tm1, x_tm0], axis=1), self.filters)
        #x_t = T.dot(x_tm0, self.filters)

        h_tm1_dot_u = T.dot(h_tm1, u)
        gates_fiom = x_t + h_tm1_dot_u + self.b

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

    def _compute_x_dot_w(self, dropout_active):
        X = self.l_in.output(dropout_active=dropout_active)
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
            dropout_corr = 1.0
        else:
            dropout_corr = 1.0 - self.p_drop
        #x_dot_w = T.dot(X, self.w * dropout_corr) + self.b
        #return x_dot_w
        return X

    def _reverse_if_backward(self, cells, out):
        if self.backward:
            out = out[::-1, ]
            cells = cells[::-1, ]
        return cells, out

    def _prepare_result(self, cells, out):
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

    def _prepare_outputs_info(self, x_dot_w):
        outputs_info = [
            T.repeat(self.init_c.dimshuffle('x', 0), x_dot_w.shape[1], axis=0),
            T.repeat(self.init_h.dimshuffle('x', 0), x_dot_w.shape[1], axis=0),
        ]
        return outputs_info

    def _process_scan_output(self, res):
        (out, cells), _ = res

        return out, cells

    def _compute_seq(self, x_dot_w, dropout_active):
        outputs_info = self._prepare_outputs_info(x_dot_w)
        x_dot_w = T.concatenate(
            [
                T.zeros((2, x_dot_w.shape[1], x_dot_w.shape[2])),
                x_dot_w
            ]
        )

        res = theano.scan(self.step,
                                      sequences=[
                                          dict(input=x_dot_w, taps=[-2, -1, 0])
                                      ],
                                      outputs_info=outputs_info,
                                      non_sequences=[self.u, self.p_vec_f,
                                                     self.p_vec_i,
                                                     self.p_vec_o,
                                                     1 if dropout_active else
                                                     0],
                                      truncate_gradient=self.truncate_gradient,
                                      go_backwards=self.backward
        )
        out, cells = self._process_scan_output(res)
        return cells, out

    def output(self, dropout_active=False):
        x_dot_w = self._compute_x_dot_w(dropout_active)

        cells, out = self._compute_seq(x_dot_w, dropout_active)
        cells, out = self._reverse_if_backward(cells, out)

        return self._prepare_result(cells, out)

    def get_params(self):
        return self.l_in.get_params().union(self.params)


class TurnSquasher(Layer):
    def __init__(self, n_features):
        self.n_features = n_features

    def connect(self, prev_layer, scores, y_time, y_seq_id):
        self.prev_layer = prev_layer
        self.scores = scores
        self.y_time = y_time
        self.y_seq_id = y_seq_id

        self.size = self.n_features

    def output(self, dropout_active=False):
        out = self.prev_layer.output(dropout_active=dropout_active)

        res, _ = theano.scan(self._step,
                          sequences=[self.y_time, self.y_seq_id],
                          non_sequences=[out]
        )
        return res

    def _step(self, y_t, y_seq_id, x):
        ivec = T.zeros((self.n_features,))
        res = x[0:y_t + 1, y_seq_id]
        scores = self.scores[0:y_t + 1, y_seq_id]
        res = theano.tensor.set_subtensor(ivec[res], scores)
        return res

    def get_params(self):
        return self.prev_layer.get_params()


class TurnSplitter(Layer):
    def __init__(self, n_features):
        self.n_features = n_features

    def connect(self, prev_layer, bounds):
        self.prev_layer = prev_layer
        self.bounds = bounds

        self.size = self.n_features

    def output(self, dropout_active=False):
        out = self.prev_layer.output(dropout_active=dropout_active)

        #y_time_ex = T.concatenate([[0], self.y_time])
        #y_seq_id_ex = T.concatenate([[0], self.y_seq_id])

        res, _ = theano.scan(self._step,
                          sequences=[dict(input=self.bounds, taps=[-1, 0])],
                          non_sequences=[out]
        )
        return res

    def _step(self, b_tm1, b_t, x):
        ivec = T.zeros((self.n_features,))
        ndxs = T.arange(b_t.shape[0])
        res = x[b_tm1:b_t + 1, ndxs]
        res = theano.tensor.set_subtensor(ivec[res], 1.0)
        return res

    def get_params(self):
        return self.prev_layer.get_params()