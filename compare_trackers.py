from collections import defaultdict
import json
import os

import numpy as np


from data import Data


def load_labels(flist_path, flist_root):
    labels = {}
    with open(flist_path) as f_in:
        for ln in f_in:
            data = json.load(open(os.path.join(flist_root, ln.strip(),
                                               'label.json')))
            labels[data['session-id']] = data

    return labels


def get_best_hypothesis(turn):
    food_vals = turn['goal-labels'].get('food', {'_null_': 1.0})
    if not len(food_vals):
        food_vals['_null_'] = 1.0

    v, p = sorted(food_vals.items(), key=lambda x: x[1])[-1]
    p_mass = sum(p for _, p in turn['goal-labels'].get('food', {}).items())
    p_null = 1.0 - p_mass
    if p > p_null:
        return v
    else:
        return '_null_'


def main(xtrack_data, tracker, dataset):
    assert len(tracker) == 2
    data = Data.load('data/xtrack/%s/%s.json' % (xtrack_data, dataset, ))
    data.vocab_rev = {val: key for key, val in data.vocab.iteritems()}

    data_map = {}
    for seq in data.sequences:
        labeling = set()
        for label in seq['labels']:
            pred_label = {}

            labeling.add(label['time'])

        texts = []
        scores = []
        curr = []
        for i, (d, ds) in enumerate(zip(seq['data'], seq['data_score'])):
            curr.append(d) #data.vocab_rev[d])
            if i in labeling:
                texts.append((np.exp(ds[0]), " ".join(data.vocab_rev[i[0]] for i in curr), seq['tags']))
                #texts.append((0.0, " ".join(curr)))
                curr = []
        assert len(curr) == 0
        #print seq['id']
        data_map[seq['id']] = texts

    flist_path = 'data/dstc2/scripts/config/dstc2_%s.flist' % dataset
    #flist_path = '/xdisk/data/dstc/dstc2/scripts/config/dstc2_test.flist'
    flist_root = 'data/dstc2/data/'

    labels = load_labels(flist_path, flist_root)

    tracker_data = []
    for t_file in tracker:
        tracker_data.append(json.load(open(t_file)))

    sess_map = defaultdict(dict)
    sess_ids = set()
    for s1, s2 in zip(tracker_data[0]['sessions'], tracker_data[1][
        'sessions']):
        sess_map[0][s1['session-id']] = s1
        sess_map[1][s2['session-id']] = s2

        sess_ids.add(s1['session-id'])
        sess_ids.add(s2['session-id'])

    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats1 = defaultdict(int)
    stats2 = defaultdict(int)

    acc_good = 0.0
    acc_total = 0.0
    acc_good_our = 0.0
    acc_total_our = 0.0
    for id in sess_ids:
        print '>', id
        texts =  data_map[id]
        s1, s2 = sess_map[0][id], sess_map[1][id]
        lbl = labels[id]
        for (score, text, tags), turn1, turn2, turn_true in \
            zip(texts, s1['turns'], s2['turns'], lbl['turns']):
            print ">> Turn:"
            print "  %8.2f  " % score, text #
            print "                 tags:", tags.get('food')
            print "           true input: %s" % turn_true['transcription']
            food_d1 = get_best_hypothesis(turn1)
            food_d2 = get_best_hypothesis(turn2)
            food_t = turn_true['goal-labels'].get('food', '_null_')



            a1 = food_d1
            a2 = food_d2
            flags = ""
            if not (a1 == a2):
                flags += '  BAD'
            else:
                flags += '  GOOD'

            if (a2 == food_t):
                flags += ' CORRECT'
            else:
                flags += ' INCORRECT'

            if (a1 == food_t):
                flags += ' BCORRECT'
            else:
                flags += ' BINCORRECT'

            print flags


            if a1 == food_t:
                acc_good += 1
            acc_total += 1

            if a2 == food_t:
                acc_good_our += 1
            acc_total_our += 1

            if a1 != '_null_' or a2 != '_null_' or food_t != '_null_':
                print '    bas: %.30s our: %.30s true: %s' % (food_d1,
                                                          food_d2, food_t)
                print

            stats[food_t][food_d1][food_d2] += 1
            stats1[food_d1] += 1
            stats2[food_d2] += 1

    res = []
    for food in stats:
        n_baseline_good = sum(stats[food][food].values())
        n_baseline_total = stats1[food]

        n_our_good = 0
        for food2 in stats[food]:
            n_our_good += stats[food][food2][food]

        n_our_total = stats2[food]

        res.append((food, n_baseline_good * 1.0 / n_baseline_total, n_our_good * 1.0 / n_our_total, n_our_total ))

    res.sort(key=lambda x: x[-1])

    for x in res:
        print '%30s: bas(%.2f) our(%.2f) our_total(%d)' % x

    print 'baseline accuracy', acc_good / acc_total
    print 'our accuracy', acc_good_our / acc_total_our


if __name__ == '__main__':
    from utils import pdb_on_error
    pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('xtrack_data')
    parser.add_argument('--tracker', action='append')

    args = parser.parse_args()

    main(**vars(args))