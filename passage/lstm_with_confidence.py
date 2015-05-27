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
from layers import Layer, dropout

import numpy as np


class LstmWithConfidence(Layer):

    def __init__(self, name=None, size=256, init=inits.normal, truncate_gradient=-1,
                 seq_output=False, p_drop=0., init_scale=0.1, out_cells=False,
                 peepholes=False, enable_branch_exp=False, backward=False,
                 learn_init_state=True, update_thresh=0.0):
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
        self.update_thresh = update_thresh

    def _init_input_connections(self, n_in):
        self.w = self.init((n_in, self.size * 4),
                           fan_in=n_in,
                           name=self._name_param("W"))
        self.b = inits.const((self.size * 4, ),
                             val=0.1,
                             name=self._name_param("b"))

        #self.br = self.init((self.size * 4, ),
        #                   layer_width=self.size,
        #                   scale=self.init_scale,
        #                   name=self._name_param("br"))

        # Initialize forget gates to large values.
        #b = self.b.get_value()
        #b[:self.size] = np.random.uniform(low=40.0, high=50.0, size=self.size)
        #b[self.size:] = 0.0
        #self.b.set_value(b)
        self.conf_a = inits.const((1, ),
                                 val=1.0,
                                 name=self._name_param("conf_a"))[0]

        self.conf_b = inits.const((1, ),
                                  val=0.0,
                                  name=self._name_param("conf_a"))[0]

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

    def connect(self, l_in, x_conf, init_c=None, init_h=None):
        self.l_in = l_in
        self.x_conf = x_conf

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

    def step(self, x_t, x_conf, h_tm1, c_tm1, u, p_vec_f, p_vec_i, p_vec_o, conf_a, conf_b,
             dropout_active):
        h_tm1_dot_u = T.dot(h_tm1, u)
        gates_fiom = x_t + h_tm1_dot_u

        g_f = self._slice(gates_fiom, 0)
        g_i = self._slice(gates_fiom, 1)
        g_m = self._slice(gates_fiom, 3)

        if self.peepholes:
            g_f += c_tm1 * p_vec_f
            g_i += c_tm1 * p_vec_i

        #g_i += (conf_a * x_conf + conf_b).dimshuffle(0, 'x')

        g_f = self.gate_act(g_f)
        g_i = self.gate_act(g_i)
        g_m = self.modul_act(g_m)

        c_t = g_f * c_tm1 + g_i * g_m

        g_o = self._slice(gates_fiom, 2)

        if self.peepholes:
            g_o += c_t * p_vec_o

        g_o = self.gate_act(g_o)

        h_t = g_o * T.tanh(c_t)

        update = T.gt(x_conf, self.update_thresh).dimshuffle(0, 'x')

        h_t = update * h_t + (1 - update) * h_tm1
        c_t = update * c_t + (1 - update) * c_tm1

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
                                      sequences=[x_dot_w, self.x_conf],
                                      outputs_info=outputs_info,
                                      non_sequences=[self.u, self.p_vec_f,
                                                     self.p_vec_i,
                                                     self.p_vec_o,
                                                     self.conf_a, self.conf_b,
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