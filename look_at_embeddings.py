import sys
import cPickle

import data

f_in = sys.argv[1]

embs = cPickle.load(open(f_in))['model_params']['emb__emb']

xd = data.Data.load('data/xtrack/e2_tagged/train.json')

for word, word_ndx in xd.vocab.iteritems():
    print word

