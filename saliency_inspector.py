import collections
import json
import numpy as np

from model import Model
from data import Data


def main(mode, model_file, data_file, json_out):
    model = Model.load(model_file, build_train=False)
    data = Data.load(data_file)

    if mode == 'inspect':
        inspect(model, data, json_out)
    elif mode == 'gen':
        generate(model)


def _build_saliency_dist(embds, explain):
    word_dist = []
    for word_id, emb in enumerate(embds):
        sal = np.dot(explain[-1, 0, :], emb)
        word_dist.append(sal)
    return word_dist

#def viterbi(trellis):
#    #curr = np.zeros_like(trellis[0])

#    #for t in trellis:


def generate(model):
    inv_cls = {val: key for key, val in model.slot_classes['food'].iteritems()}
    embds = model.input_emb.get_value()
    vocab_rev = {val: key for key, val in model.vocab.iteritems()}

    good = collections.defaultdict(int)
    total = 0

    for cls_ndx, cls in inv_cls.iteritems():
        print '# cls', cls
        dlg = []
        cls_ndx = 18
        for i in range(100):
            i = np.zeros((7, len(vocab_rev), ))
            explain = model.fi(np.dot(i, embds)[:, np.newaxis, np.newaxis, :], [0] * 7, [0, 1, 2, 3, 4, 5, 6], [cls_ndx] * 7)
            new_i = np.dot(explain[:, 0, 0, :], embds.T)
            i = i + 0.01 * new_i
            i[i < 0.0] = 0.0


        import ipdb; ipdb.set_trace()


        for i in range(7):
            in_dlg = np.array(dlg + [0])[:,np.newaxis, np.newaxis].astype('int32')
            explain = model.f(embds[in_dlg], [0], [len(in_dlg) - 1], [cls_ndx])

            #i += explain

            #import ipdb; ipdb.set_trace()
            #word_dist = _build_saliency_dist(embds, explain)
            word_dist = np.dot(explain[-1, 0, :], embds.T).argsort()[0]
            word_dist_lex = [vocab_rev[x] for x in word_dist]

            #trellis.append(word_dist)
            #for word_id in word_dist:
            #    print '  ', vocab_rev[word_id]
            #print 'sal', list(word_dist)

            #if cls == 'chinese':
            #    import ipdb; ipdb.set_trace()

            best_word = word_dist[0]
            in_dlg = np.array(dlg + [best_word])[:,np.newaxis, np.newaxis].astype('int32')
            pred_score = model._predict(in_dlg, [0], [len(in_dlg) - 1])[0][0][cls_ndx]
            print '  ', "%.2f" % pred_score,  vocab_rev[best_word]

            cls_word = cls.split()[0]
            try:
                good[word_dist_lex.index(cls_word)] += 1
            except ValueError:
                good[-1] += 1

            total += 1
            dlg.append(best_word)

        #for words in zip(*trellis):
        #    print '  ', "\t".join(vocab_rev[w] for w in words)
        print ''
        #import ipdb; ipdb.set_trace()

    import ipdb; ipdb.set_trace()


def inspect(model, data, out_json):
    valid_data = model.prepare_data_predict(data.sequences, ['food'])
    valid_data_trn = model.prepare_data_train(data.sequences, ['food'])
    preds = model._predict(*valid_data)
    pred = preds[0]
    labels = valid_data_trn[3]

    inv_cls = {val: key for key, val in model.slot_classes['food'].iteritems()}
    embds = model.input_emb.get_value()
    vocab_rev = {val: key for key, val in model.vocab.iteritems()}

    pred_inv = pred.argmin(axis=1)
    pred_inv_score = pred.min(axis=1)
    pred_sort = pred.argsort(axis=1)
    pred_max = pred.argmax(axis=1)
    pred_max_score = pred.max(axis=1)

    x = valid_data[0]

    res = []

    for (seq_id, seq_time), y_inv, y_inv_score, y_sort, lbl, lbl_pred, lbl_pred_score in zip(zip(*valid_data[1:]), pred_inv, pred_inv_score, pred_sort, labels, pred_max, pred_max_score):
        for y_inv in [y_inv]: #inv_cls:
            print '# seq(%d) until_time(%d) lbl(%s) pred(%s; %.2f) pred_inv(%s; %.2f)' % (seq_id, seq_time, inv_cls[lbl], inv_cls[lbl_pred], lbl_pred_score, inv_cls[y_inv], y_inv_score )
            dlg = x[:seq_time + 1, seq_id:seq_id+1, :]
            dlg_seq_id = [0]
            dlg_time = [seq_time]
            #explain = model.f(dlg, dlg_seq_id, dlg_time, [y_inv])
            explain = model.f(dlg, dlg_seq_id, dlg_time, [y_inv])
            explain2 = model.f2(dlg, dlg_seq_id, dlg_time, [lbl_pred])

            #explain3 = model.f(dlg, dlg_seq_id, dlg_time, [lbl_pred])

            words = []
            saliences = []
            saliences2 = []

            new_item = []
            for i, (t, et, et2) in enumerate(zip(dlg, explain, explain2)):
                print '   . ', 't=%d' % i
                for w, ew, ew2 in zip(t[0, :], et[0,:], et2[0,:]):
                    sal = np.dot(embds[w], ew)
                    sal2 = np.dot(embds[w], ew2)
                    #print '     %35s sal(%10.4f) sal2(%10.4f) errnorm(%10.4f) embnorm(%10.4f)' % (vocab_rev[w], sal, sal2, np.linalg.norm(ew, 2), np.linalg.norm(embds[w], 2))
                    new_item.append((vocab_rev[w], sal, sal2, list(ew), list(ew2)))

                    words.append(vocab_rev[w])
                    saliences.append(sal)
                    saliences2.append(sal2)

            saliences = np.array(saliences)
            saliences2 = np.array(saliences2)

            saliences = saliences / saliences.std()
            saliences2 = saliences2 / saliences2.std()

            for word, sal1, sal2 in zip(words, saliences, saliences2):
                print '    %35s      %.2f     %.2f' % (word, sal1, sal2, )



            res.append(dict(seq_id=seq_id, until_time=seq_time, lbl=inv_cls[lbl], pred_inv=inv_cls[y_inv], pred=inv_cls[lbl_pred], data=new_item))

    with open(out_json, 'w') as f_out:
        json.dump(res, f_out)





if __name__ == '__main__':
    import utils
    utils.pdb_on_error()
    utils.init_logging('SaliencyInspector')
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='inspect')
    parser.add_argument('--json_out', default='inspect.json')
    parser.add_argument('model_file')
    parser.add_argument('data_file')

    args = parser.parse_args()

    main(**vars(args))