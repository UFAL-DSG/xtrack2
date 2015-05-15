from collections import defaultdict

import numpy as np

def main(fname):
    with open(fname) as f_in:
        data = [ln.split('\t') for ln in f_in.read().split('\n') if len(ln.strip()) > 0]
        goals_ndx = data[0].index('a_best_acc_food')
        #goals_ndx = data[0].index('a_best_acc_goals')
        eid_ndx = data[0].index('eid')

        res = []
        for ln in data[1:]:
            val = float(ln[goals_ndx])
            eid = ln[eid_ndx]
            res.append((eid, val, ))

        for a_eid, val, val_med, val_std, val_min, val_max in aggregate(res):
            print "%30smed(%8.2f) std(%8.2f) max(%8.2f)" % (a_eid, val_med, val_std, val_max)


def aggregate(vals):
    res = defaultdict(list)
    for eid, val in vals:
        res[extract_a_eid(eid)].append(val)

    res2 = []
    for key, rvals in res.iteritems():
        res2.append((key, np.mean(rvals), np.median(rvals), np.std(rvals), np.min(rvals), np.max(rvals)))

    return sorted(res2)


def extract_a_eid(eid):
    return eid[:eid.rindex('_')]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fname')

    args = parser.parse_args()

    main(**vars(args))
