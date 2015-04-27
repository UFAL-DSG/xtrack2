import numpy as np

from model import Model
from data import Data


def main(model_file, data_file):
    model = Model.load(model_file, build_train=False)
    data = Data.load(data_file)

    inspect(model, data)


def inspect(model, data):
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

    for (seq_id, seq_time), y_inv, y_inv_score, y_sort, lbl, lbl_pred, lbl_pred_score in zip(zip(*valid_data[1:]), pred_inv, pred_inv_score, pred_sort, labels, pred_max, pred_max_score):
        print '# seq(%d) until_time(%d) lbl(%s) pred(%s; %.2f) pred_inv(%s; %.2f)' % (seq_id, seq_time, inv_cls[lbl], inv_cls[lbl_pred], lbl_pred_score, inv_cls[y_inv], y_inv_score )
        dlg = x[:seq_time + 1, seq_id:seq_id+1, :]
        dlg_seq_id = [0]
        dlg_time = [seq_time]
        #explain = model.f(dlg, dlg_seq_id, dlg_time, [y_inv])
        explain = model.f(dlg, dlg_seq_id, dlg_time, [y_inv])
        explain2 = model.f2(dlg, dlg_seq_id, dlg_time, [lbl_pred])

        for i, (t, et, et2) in enumerate(zip(dlg, explain, explain2)):
            print '   . ', 't=%d' % i
            for w, ew, ew2 in zip(t[0, :], et[0,:], et2[0,:]):
                sal = np.dot(embds[w], ew)
                sal2 = np.dot(embds[w], ew2)
                print '     %35s sal(%10.4f) sal2(%10.4f) errnorm(%10.4f) embnorm(%10.4f)' % (vocab_rev[w], sal, sal2, np.linalg.norm(ew, 2), np.linalg.norm(embds[w], 2))



if __name__ == '__main__':
    import utils
    utils.pdb_on_error()
    utils.init_logging('SaliencyInspector')
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('data_file')

    args = parser.parse_args()

    main(**vars(args))