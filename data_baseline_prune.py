import os
from collections import Counter
from itertools import count
import json


def main(dataset, fixed_dataset, cutoff_thresh):
    prunner = DataBaselinePrunner(cutoff_thresh)
    prunner.prune(dataset, fixed_dataset)


class DataBaselinePrunner:
    def __init__(self, cutoff_thresh):
        self.cutoff_thresh = cutoff_thresh

    def prune(self, dataset, fixed_dataset):
        data = self._load_data(dataset)
        data_fixed = self._load_data(fixed_dataset)

        intersect_ftrs = self._get_common_good_features(data)

        self._prune_non_selected_features(data, intersect_ftrs)
        self._prune_non_selected_features(data_fixed, intersect_ftrs)

        ftrs = {ftr: i for ftr, i in zip(intersect_ftrs, count())}
        print len(ftrs)
        self._save_data(data, dataset, ftrs)
        self._save_data(data_fixed, fixed_dataset, ftrs)

    def _load_data(self, dataset):
        data = []
        for data_file in dataset:
            with open(data_file) as f_in:
                data.append(json.load(f_in))
        return data

    def _get_common_good_features(self, data):
        ftrs = []
        for d in data:
            ftrs.append(self._get_good_features(d))
        intersect_ftrs = reduce(set.intersection, ftrs)
        return intersect_ftrs


    def _get_good_features(self, d):
        ftr_cnts = Counter()
        for seq in d['sequences']:
            for ftrs in seq['data']:
                for ftr in ftrs:
                    ftr_cnts[ftr] += 1

        good_features = set()
        for key, val in ftr_cnts.iteritems():
            if val >= self.cutoff_thresh:
                good_features.add(key)

        return good_features

    def _prune_non_selected_features(self, data, intersect_ftrs):
        for d in data:
            for seq in d['sequences']:
                for ftrs in seq['data']:
                    for ftr in list(ftrs):
                        if not ftr in intersect_ftrs:
                            del ftrs[ftr]

    def _save_data(self, data, dataset, ftrs):
        for d, dataset_fn in zip(data, dataset):
            d['vocab'] = ftrs
            print 'writing', len(ftrs), dataset_fn
            with open(dataset_fn, 'w') as f_out:
                json.dump(d, f_out, indent=4, sort_keys=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='append')
    parser.add_argument('--fixed_dataset', action='append')
    parser.add_argument('--cutoff_thresh', type=int, required=True)

    args = parser.parse_args()

    main(**vars(args))