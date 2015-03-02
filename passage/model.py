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
    def __init__(self):
        self.params = []

    def save_params(self, f_name):
        model_params = {}
        for param in self.params:
            model_params[param.name] = param.get_value()

        with open(f_name, 'w') as f_out:
            cPickle.dump(model_params, f_out, -1)

    def load_params(self, f_name):
        with open(f_name) as f_in:
            model_params = cPickle.load(f_in)

        for param in self.params:
            param_val = model_params.get(param.name)
            if param_val != None:
                param.set_value(param_val)

    def save(self, f_name):
        val = sys.getrecursionlimit()
        sys.setrecursionlimit(100000)
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
