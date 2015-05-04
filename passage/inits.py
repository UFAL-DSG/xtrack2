import numpy as np

import theano
import theano.tensor as T

from utils import sharedX, floatX, intX



def uniform(shape, layer_width, scale=0.1, name=None):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape),
                   name=name)

def normal(shape, layer_width, scale=0.1, name=None):
    scale = scale / np.sqrt(layer_width)
    return sharedX(np.random.randn(*shape) * scale, name=name)


def normal_relu(shape, layer_width, scale=0.1, name=None):
    scale = np.sqrt(2.0 / layer_width)
    return sharedX(np.random.randn(*shape) * scale, name=name)


def copy(params):
    def _copy(*args, **kwargs):
        name = kwargs.pop('name')
        for param in params:
            if param.name == name:
                return param

        raise Exception('Param to copy not found.')

    return _copy

"""
def orthogonal(shape, scale=1.1):
    " benanne lasagne ortho init (faster than qr approach)"
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])
"""