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
    pred_max = pred.argmax(axis=1)
    x = valid_data[0]

    for (seq_id, seq_time), y_inv, lbl, lbl_pred in zip(zip(*valid_data[1:]), pred_inv, labels, pred_max):
        print '# seq(%d) until_time(%d) lbl(%s) pred(%s)' % (seq_id, seq_time, inv_cls[lbl], inv_cls[lbl_pred] )
        dlg = x[:seq_time + 1, seq_id:seq_id+1, :]
        dlg_seq_id = [0]
        dlg_time = [seq_time]
        explain = model.f(dlg, dlg_seq_id, dlg_time, [y_inv])

        for i, (t, et) in enumerate(zip(dlg, explain)):
            print '   . ', 't=%d' % i
            for w, ew in zip(t[0, :], et[0,:]):
                sal = np.dot(embds[w], ew)
                print '     %15s %10.4f' % (vocab_rev[w], sal, )



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