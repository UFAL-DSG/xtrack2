import cPickle
import sys
import theano
import theano.tensor as T
import numpy as np
from time import time

import costs
import updates
import iterators 
from iterators import iter_data, padded
from utils import case_insensitive_import
from preprocessing import LenFilter, standardize_targets

def flatten(l):
    return [item for sublist in l for item in sublist]



class NeuralModel(object):
    def save(self, f_name):
        val = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        with open(f_name, 'w') as f_out:
            cPickle.dump(self, f_out, -1)

        sys.setrecursionlimit(val)

    @classmethod
    def load(cls, f_name):
        orig = theano.config.reoptimize_unpickled_function
        theano.config.reoptimize_unpickled_function = False

        with open(f_name) as f_in:
            res = cPickle.load(f_in)

        theano.config.reoptimize_unpickled_function = orig
        return res
