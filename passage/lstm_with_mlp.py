class LstmWithMLP(LstmRecurrent):

    def __init__(self, name=None, size=256, init=inits.normal, truncate_gradient=-1,
                 seq_output=False, p_drop=0., init_scale=0.1, out_cells=False,
                 peepholes=False, enable_branch_exp=False, backward=False,
                 mlps=None):
        super(LstmWithMLP, self).__init__(name, size, init, truncate_gradient,
                                          seq_output, p_drop, init_scale,
                                          out_cells, peepholes,
                                          enable_branch_exp, backward)
        self.mlps = mlps

    def connect(self, l_in):
        super(LstmWithMLP, self).connect(l_in)

        self.mlp_inits = []
        n_mlp_inputs = 0
        for i, mlp in enumerate(self.mlps):
            mlp_init = self.init((mlp.size, ), layer_width=mlp.size,
                                 name=self._name_param("mlp_init_%d" % i))
            self.mlp_inits.append(mlp_init)
            n_mlp_inputs += mlp.size

        self.params.extend(self.mlp_inits)

        self.w_mlp = self.init((n_mlp_inputs, self.size * 4),
                           layer_width=self.size,
                           scale=self.init_scale,
                           name=self._name_param("Wmlp"))
        self.params.append(self.w_mlp)

    def _prepare_outputs_info(self, x_dot_w):
        outputs_info = super(LstmWithMLP, self)._prepare_outputs_info(x_dot_w)
        for mlp_init in self.mlp_inits:
            outputs_info.append(
                T.repeat(mlp_init.dimshuffle('x', 0), x_dot_w.shape[1],
                         axis=0),
            )
        return outputs_info

    def _compute_seq(self, x_dot_w, dropout_active):
        res = super(LstmWithMLP, self)._compute_seq(x_dot_w, dropout_active)
        res = list(res)
        h_t = res.pop(0)
        c_t = res.pop(0)
        return h_t, c_t

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
        mlp_tm1 = []
        for mlp in self.mlps:
            mlp_tm1.append(args.pop(0))
        xmlp_t_concat = T.concatenate(mlp_tm1, axis=1)
        x_t += T.dot(xmlp_t_concat, self.w_mlp)

        u = args.pop(0)
        p_vec_f = args.pop(0)
        p_vec_i = args.pop(0)
        p_vec_o = args.pop(0)
        dropout_active = args.pop(0)

        h_t, c_t = super(LstmWithMLP, self).step(x_t, h_tm1, c_tm1, u, p_vec_f,
                                                 p_vec_i, p_vec_o, dropout_active)

        outs = [h_t, c_t]

        #h_t_layer = IdentityInput(h_t, self.size)

        for mlp in self.mlps:
            #mlp.connect(h_t_layer)
            mlp.l_in.set_val(h_t)
            outs.append(mlp.output(dropout_active=bool(dropout_active)))

        return tuple(outs)