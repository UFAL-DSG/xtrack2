import theano.tensor as T
import theano

from layers import LstmRecurrent
import inits
import activations

class LstmWithMLP(LstmRecurrent):

    def __init__(self, name=None, size=256, init=inits.normal, truncate_gradient=-1,
                 seq_output=False, p_drop=0., init_scale=0.1, out_cells=False,
                 peepholes=False, enable_branch_exp=False, backward=False,
                 mlp_n_classes=-1, mlp_n_layers=-1, mlp_n_hidden=-1):
        super(LstmWithMLP, self).__init__(name, size, init, truncate_gradient,
                                          seq_output, p_drop, init_scale,
                                          out_cells, peepholes,
                                          enable_branch_exp, backward)

        assert mlp_n_layers > 0

        self.mlp_n_classes = mlp_n_classes
        self.mlp_n_layers = mlp_n_layers
        self.mlp_n_hidden = mlp_n_hidden

        self.mlp_params = []

        for l_id in range(self.mlp_n_layers):
            if l_id == 0:
                n_in = self.size
            else:
                n_in = self.mlp_n_hidden

            if l_id == self.mlp_n_layers - 1:
                n_out = self.mlp_n_classes
                act = activations.softmax
            else:
                n_out = self.mlp_n_hidden
                act = activations.rectify

            print n_out

            w = self.init((n_in, n_out), fan_in=n_in, name=self._name_param("mlp_w%d" % l_id ))
            b = inits.const((n_out, ), val=0.1, name=self._name_param("mlp_b%d" % l_id))

            self.mlp_params.append((act, w, b))



    def connect(self, l_in):
        super(LstmWithMLP, self).connect(l_in)

        self.mlp_init = self.init((self.mlp_n_classes, ), fan_in=self.mlp_n_classes,
                             name=self._name_param("mlp_init"))

        self.params.append(self.mlp_init)

        self.w_mlp_to_gates = self.init((self.mlp_n_classes, self.size * 4),
                           fan_in=self.mlp_n_classes,
                           name=self._name_param("Wmlp"))
        self.params.append(self.w_mlp_to_gates)

        for _, w, b in self.mlp_params:
            self.params.append(w)
            self.params.append(b)

    def _prepare_outputs_info(self, x_dot_w):
        outputs_info = super(LstmWithMLP, self)._prepare_outputs_info(x_dot_w)
        outputs_info.append(
            T.repeat(self.mlp_init.dimshuffle('x', 0), x_dot_w.shape[1],
                     axis=0),
        )

        return outputs_info

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
        out, cells, mlp = res[0]
        return cells, out, mlp

    def _process_scan_output(self, res):
        out = res[0][0]
        cells = res[0][1]

        return out, cells

    def step(self, *args):
        i = 0
        args = list(args)

        x_t = args.pop(0)
        h_tm1 = args.pop(0)
        c_tm1 = args.pop(0)
        xmlp_tm1 = args.pop(0)
        x_t += T.dot(xmlp_tm1, self.w_mlp_to_gates)

        u = args.pop(0)
        p_vec_f = args.pop(0)
        p_vec_i = args.pop(0)
        p_vec_o = args.pop(0)
        dropout_active = args.pop(0)

        h_t, c_t = super(LstmWithMLP, self).step(x_t, h_tm1, c_tm1, u, p_vec_f,
                                                 p_vec_i, p_vec_o, dropout_active)

        outs = [h_t, c_t]

        curr = h_t
        for act, w, b in self.mlp_params:
            curr = act(T.dot(curr, w) + b)

        outs.append(curr)

        return tuple(outs)

    def output(self, dropout_active=False):
        x_dot_w = self._compute_x_dot_w(dropout_active)

        cells, out, mlp = self._compute_seq(x_dot_w, dropout_active)

        return mlp