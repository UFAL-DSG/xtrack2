from collections import defaultdict

import numpy as np
from scipy import stats

def main(fname, slot):
    with open(fname) as f_in:
        data = [ln.split('\t') for ln in f_in.read().split('\n') if len(ln.strip()) > 0]
        goals_ndx = data[0].index('a_best_acc_%s' % slot)
        #goals_ndx = data[0].index('a_best_acc_goals')
        eid_ndx = data[0].index('eid')

        res = []
        for ln in data[1:]:
            val = float(ln[goals_ndx])
            eid = ln[eid_ndx]
            res.append((eid, val, ))
        
        vals_set = []
        for a_eid, vals, val, val_med, val_std, val_min, val_max in aggregate(res):
            print "%30s mean(%.2f) med(%.2f) std(%.2f) max(%.2f)" % (a_eid, val, val_med, val_std, val_max)
            vals_set.append(vals)
        
        insignificant = []
        for i, v1 in enumerate(vals_set):
            #print '%.2d: ' % i + 1,
            for y, v2 in enumerate(vals_set):
                #print "%.2f" % stats.ttest_1samp(v1, np.mean(v2))[1],
                #print "%.2f" % stats.ttest_rel(v1, v2)[1],
                #print "%.2f" % stats.ttest_ind(v1, v2, equal_var=False)[1],
                p_val = stats.ttest_ind(v1, v2, equal_var=False)[1]
                if p_val > 0.01:
                    if i < y:
                        insignificant.append((i + 1, y + 1, ))

            #print
        print 'Insignificant differences (p>0.01):', insignificant

def aggregate(vals):
    res = defaultdict(list)
    for eid, val in vals:
        res[extract_a_eid(eid)].append(val)

    res2 = []
    for key, rvals in res.iteritems():
        res2.append((key, rvals, np.mean(rvals), np.median(rvals), np.std(rvals), np.min(rvals), np.max(rvals)))

    return sorted(res2)


def extract_a_eid(eid):
    return eid[:eid.rindex('_')]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    parser.add_argument('--slot', default="goals")

    args = parser.parse_args()

    main(**vars(args))
