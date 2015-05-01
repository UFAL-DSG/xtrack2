import theano
import theano.tensor as T
import numpy as np

from utils import shared0s, floatX


def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g*c/n, g)
    return g


def clip_norms(gs, c):
    norm = T.sqrt(sum([T.sum(g**2) for g in gs]))
    return [clip_norm(g, c, norm) for g in gs]


def max_norm(p, maxnorm):
    if maxnorm > 0:
        norms = T.sqrt(T.sum(T.sqr(p)))
        desired = T.clip(norms, 0, maxnorm)
        p = p * (desired/ (1e-7 + norms))
    return p

class Regularizer(object):

    def __init__(self, l1=0., l2=0., maxnorm=0.):
        self.l1 = l1
        self.l2 = l2
        self.maxnorm = maxnorm

    def gradient_regularize(self, p, g):
        if self.l1 > 0 or self.l2 > 0:
            g += p * self.l2
            g += T.sgn(p) * self.l1
        return g

    def weight_regularize(self, p):
        p = max_norm(p, self.maxnorm)
        return p


class Update(object):

    def __init__(self, regularizer=Regularizer(), clipnorm=0.):
        self.regularizer = regularizer
        self.clipnorm = clipnorm

    def get_updates(self, params, cost):
        raise NotImplementedError

    def get_update_ratio(self, params, updates):
        res = 0.0
        for p, np in updates:
            if p in params:
                res += (np - p).norm(2) / p.norm(2)
        res = res / len(updates)
        return res


class SGD(Update):

    def __init__(self, lr=0.01, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.lr = lr

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p, g in zip(params, grads):
            #g = self.regularizer.gradient_regularize(p, g)
            updated_p = p - self.lr * g
            #updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, T.cast(updated_p, dtype=theano.config.floatX)))

        return updates


class Momentum(Update):

    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.lr = lr
        self.momentum = momentum

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            m = theano.shared(p.get_value() * 0.)
            v = (self.momentum * m) - (self.lr * g)
            updates.append((m, v))

            updated_p = p + v
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates


class NAG(Update):

    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.lr = lr
        self.momentum = momentum

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            m = theano.shared(p.get_value() * 0.)
            v = (self.momentum * m) - (self.lr * g)
            updates.append((m,v))

            updated_p = p + self.momentum * v - self.lr * self.regularizer.gradient_regularize(p, g)
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates


class RMSprop(Update):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            updates.append((acc, acc_new))

            updated_p = p - self.lr * (g / T.sqrt(acc_new + self.epsilon))
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates

class Adam(Update):

    def __init__(self, lr=0.0002, b1=0.1, b2=0.001, e=1e-8, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        i = theano.shared(floatX(0.))
        i_t = i + 1.
        fix1 = 1. - self.b1**(i_t)
        fix2 = 1. - self.b2**(i_t)
        lr_t = self.lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (self.b1 * g) + ((1. - self.b1) * m)
            v_t = (self.b2 * T.sqr(g)) + ((1. - self.b2) * v)
            g_t = m_t / (T.sqrt(v_t) + self.e)
            g_t = self.regularizer.gradient_regularize(p, g_t)
            p_t = p - (lr_t * g_t)
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates

class Adagrad(Update):

    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.lr = lr
        self.epsilon = epsilon

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.)
            acc_t = acc + g ** 2
            updates.append((acc, acc_t))

            p_t = p - (self.lr / T.sqrt(acc_t + self.epsilon)) * g
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((p, p_t))
        return updates  

class Adadelta(Update):

    def __init__(self, lr=0.5, rho=0.95, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)

            acc = theano.shared(p.get_value() * 0.)
            acc_delta = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            updates.append((acc,acc_new))

            update = g * T.sqrt(acc_delta + self.epsilon) / T.sqrt(acc_new + self.epsilon)
            updated_p = p - self.lr * update
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))

            acc_delta_new = self.rho * acc_delta + (1 - self.rho) * update ** 2
            updates.append((acc_delta,acc_delta_new))
        return updates

class RProp(Update):
    def __init__(self, lr=0.1, plus=1.4, minus=0.5, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.lr = lr
        self.plus = plus
        self.minus = minus

    def get_updates(self, params, cost):
        grads_rprop = []
        grads_history = []
        grads_rprop_new = []

        shapes = []

        grads = T.grad(cost, params)

        for param, grad in zip(params, grads):
            shape = param.shape.eval()
            shapes.append(shape)
            #grad = tt.grad(loss, wrt=param)
            #grads.append(grad)

            # Save gradients histories for RProp.
            grad_hist = theano.shared(param.get_value() * 0.0 + 1.0,
                                      name="rpop_hist_%s" % param)
            grads_history.append(
                grad_hist
            )

            # Create variables where rprop rates will be stored.
            grad_rprop = theano.shared(param.get_value() * 0.0 + self.lr,
                                       name="rprop_%s" % param)
            grads_rprop.append(grad_rprop)

            # Compute the new RProp coefficients.
            rprop_sign = T.sgn(grad_hist * grad)
            grad_rprop_new = grad_rprop * (
                T.eq(rprop_sign, 1) * self.plus
                + T.neq(rprop_sign, 1) * self.minus
            )
            grads_rprop_new.append(grad_rprop_new)

        updates = [
            # Update parameters according to the RProp update rule.
            (p, p - rg * T.sgn(g))
            for p, g, rg in zip(params, grads, grads_rprop_new)
        ] + [
            # Save current gradient for the next step..
            (hg, g) for hg, g in zip(
                grads_history, grads)
        ] + [
            # Save the new rprop grads.
            (rg, rg_new) for rg, rg_new in zip(
                grads_rprop, grads_rprop_new)
        ]

        return updates