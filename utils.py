import numpy as np
import sys
import time


def pdb_on_error():
    import sys

    def info(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
            sys.__excepthook__(type, value, tb)
        else:
            try:
                import ipdb as pdb
            except ImportError:
                import pdb
            import traceback
            # we are NOT in interactive mode, print the exception
            traceback.print_exception(type, value, tb)
            print
            #  then start the debugger in post-mortem mode.
            # pdb.pm() # deprecated
            pdb.post_mortem(tb) # more

    sys.excepthook = info


class Timer:
    def __init__(self):
        self.timestamp = None

    def start(self, duration):
        self.timestamp = time.time()
        self.duration = duration

    def elapsed(self):
        return self.timestamp + self.duration < time.time()


class P(object):
    def __init__(self):
        self.buff = []
        self.len_so_far = 0

    def print_out(self, what):
        swhat = str(what)
        self.buff.append(swhat)
        self.len_so_far += len(swhat)

    def tab(self, size):
        if self.len_so_far < size:
            self.print_out(' ' * (size - self.len_so_far))

    def render(self):
        return "".join(self.buff)



class ConfusionMatrix:
    """
    source: lasagne toolkit
       Simple confusion matrix class
       row is the true class, column is the predicted class
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.mat = np.zeros((n_classes,n_classes),dtype='int')

    def __str__(self):
        return np.array_str(self.mat)

    def batchAdd(self,y_true,y_pred):
        assert len(y_true) == len(y_pred)
        assert max(y_true) < self.n_classes
        assert max(y_pred) < self.n_classes
        for i in range(len(y_true)):
                self.mat[y_true[i],y_pred[i]] += 1

    def zero(self):
        self.mat.fill(0)

    def getErrors(self):
        """
        Calculate differetn error types
        :return: vetors of true postives (tp) false negatives (fn), false positives (fp) and true negatives (tn)
                 pos 0 is first class, pos 1 is second class etc.
        """
        tp = np.asarray(np.diag(self.mat).flatten(),dtype='float')
        fn = np.asarray(np.sum(self.mat, axis=1).flatten(),dtype='float') - tp
        fp = np.asarray(np.sum(self.mat, axis=0).flatten(),dtype='float') - tp
        tn = np.asarray(np.sum(self.mat)*np.ones(self.n_classes).flatten(),dtype='float') - tp - fn - fp
        return tp,fn,fp,tn

    def accuracy(self):
        """
        Calculates global accuracy
        :return: accuracy
        :example: >>> conf = ConfusionMatrix(3)
                  >>> conf.batchAdd([0,0,1],[0,0,2])
                  >>> print conf.accuracy()
        """
        tp, _, _, _ = self.getErrors()
        n_samples = np.sum(self.mat)
        return np.sum(tp) / n_samples


    def sensitivity(self):
        tp, tn, fp, fn = self.getErrors()
        res = tp / (tp + fn)
        res = res[~np.isnan(res)]
        return res

    def specificity(self):
        tp, tn, fp, fn = self.getErrors()
        res = tn / (tn + fp)
        res = res[~np.isnan(res)]
        return res

    def positivePredictiveValue(self):
        tp, tn, fp, fn = self.getErrors()
        res = tp / (tp + fp)
        res = res[~np.isnan(res)]
        return res

    def negativePredictiveValue(self):
        tp, tn, fp, fn = self.getErrors()
        res = tn / (tn + fn)
        res = res[~np.isnan(res)]
        return res

    def falsePositiveRate(self):
        tp, tn, fp, fn = self.getErrors()
        res = fp / (fp + tn)
        res = res[~np.isnan(res)]
        return res

    def falseDiscoveryRate(self):
        tp, tn, fp, fn = self.getErrors()
        res = fp / (tp + fp)
        res = res[~np.isnan(res)]
        return res

    def F1(self):
        tp, tn, fp, fn = self.getErrors()
        res = (2*tp) / (2*tp + fp + fn)
        res = res[~np.isnan(res)]
        return res

    def matthewsCorrelation(self):
        tp, tn, fp, fn = self.getErrors()
        numerator = tp*tn - fp*fn
        denominator = np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        res = numerator / denominator
        res = res[~np.isnan(res)]
        return res
    def getMat(self):
        return self.mat


def inline_print(string):
    sys.stderr.write('\r\t%s' % (string))
    sys.stderr.flush()


def init_logging(logger_name='XTrack'):
    import logging
    # Setup logging.
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    logging_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    formatter = logging.Formatter(logging_format)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logging.root = logger


def get_git_revision_hash():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])