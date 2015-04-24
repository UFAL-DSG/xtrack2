import cPickle
import numpy as np


def main(param_file):
    with open(param_file) as f_in:
        params = cPickle.load(f_in)

    embs = params['model_params']['emb__emb']
    w = params['model_params']['flstm_0__W']
    vocab_rev = {val: key for key, val in params['init_args']['vocab'].iteritems()}

    import ipdb; ipdb.set_trace()

    res = 1 / (1.0 + np.exp(-np.dot(embs, w)))

    for i in range(res.shape[1]):
        if i % 80 == 0:
            print '#' * 100

        dims_worst = res[:,i].argsort()[:5]
        dims_best = res[:,i].argsort()[-5:]

        for dims in [dims_worst, dims_best]:
            vals = res[dims,i]

            for word, score in zip(map(vocab_rev.__getitem__, dims)[::-1], map(lambda x: "%.2f" % x , vals[::-1])):
                print word + "\t",

            for word, score in zip(map(vocab_rev.__getitem__, dims)[::-1], map(lambda x: "%.2f" % x , vals[::-1])):
                print score+"\t",
            print

    #import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('param_file')

    args = parser.parse_args()

    main(**vars(args))