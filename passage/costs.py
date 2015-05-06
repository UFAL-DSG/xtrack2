import theano
import theano.tensor as T

def WeightedCategoricalCrossEntropy(y_true, y_pred, weights):
    ce =  T.nnet.categorical_crossentropy(y_pred, y_true)
    return (weights.dimshuffle('x', 0) * ce).mean()

def CategoricalCrossEntropy(y_true, y_pred):
    return T.nnet.categorical_crossentropy(y_pred, y_true).sum()

def BinaryCrossEntropy(y_true, y_pred):
    return T.nnet.binary_crossentropy(y_pred, y_true).mean()

def MeanSquaredError(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()

def MeanAbsoluteError(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean()

def SquaredHinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean()

def Hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean()

cce = CCE = CategoricalCrossEntropy
bce = BCE = BinaryCrossEntropy
mse = MSE = MeanSquaredError
mae = MAE = MeanAbsoluteError
