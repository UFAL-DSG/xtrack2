import sys
import os
import numpy as np
import cPickle

assert len(sys.argv) == 2

dir = sys.argv[1]

param_files = []
for f in os.listdir(dir):
    if f.startswith('params'):
        param_files.append(os.path.join(dir, f))



vocab = None
vocab_rev = None
data = []
for fname in param_files:
    with open(fname) as f_in:
        print 'loading', fname
        fdata = cPickle.load(f_in)

        if not vocab:
            vocab = fdata['init_args']['vocab']
            vocab_rev = {val: key for key, val in vocab.iteritems()}

        emb_mat = fdata['model_params']['emb__emb']
        data.append(emb_mat)


n_words = data[0].shape[0]
emb_size = data[0].shape[1]

data = np.array(data)

means = data.mean(axis=0)
var = (data).var(axis=0)

varnorms = [np.linalg.norm(wordvar) for wordvar in var]



for word_id in list(reversed(np.argsort(varnorms)))[:100]:
    varnorm = varnorms[word_id]
    print "%30s\t%.4f" % (vocab_rev[word_id], varnorm, )
