import numpy as np

import theano
import theano.tensor as T

from utils import sharedX, floatX, intX

init_var_cntr = 0

def uniform(shape, scale=0.05):
    global init_var_cntr
    init_var_cntr += 1
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape),
                   name="uniform_%d" % init_var_cntr)

def normal(shape, scale=0.05):
    global init_var_cntr
    init_var_cntr += 1
    scale = scale / np.sqrt(shape[0])
    return sharedX(np.random.randn(*shape) * scale, name="normal_%d" %
                                                         init_var_cntr)

def orthogonal(shape, scale=1.1):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])