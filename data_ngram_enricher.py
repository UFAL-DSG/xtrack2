import os
from collections import Counter, deque, defaultdict

from itertools import count
import json


def main(dataset, fixed_dataset, ngram, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ngram_order_lst = []
    ngram_cnt_lst = []
    for spec in ngram:
        ngram_order, ngram_cnt = map(int, spec.split(':'))
        ngram_order_lst.append(ngram_order)
        ngram_cnt_lst.append(ngram_cnt)

    enricher = DataNGramEnricher(ngram_order_lst, ngram_cnt_lst)
    enricher.enrich(dataset, fixed_dataset, outdir)


class DataNGramEnricher:
    def __init__(self, ngram_order, ngram_cnt):
        self.ngram_order = ngram_order
        self.ngram_cnt = ngram_cnt


    def enrich(self, dataset, fixed_dataset, outdir):
        data = self._load_data(dataset)
        data_fixed = self._load_data(fixed_dataset)

        self.enrich_data(data, data_fixed)

        self._save_data(data, dataset, outdir)
        self._save_data(data_fixed, fixed_dataset, outdir)

    def enrich_data(self, data, data_fixed):

        ngrams = self._get_ngrams(data)
        ngram_first_index = len(data[0]['vocab'])
        ngram_map = {ngram: i + ngram_first_index for i, ngram  in enumerate(ngrams)}

        self._add_ngrams(data, ngram_map)
        self._add_ngrams(data_fixed, ngram_map)

        self._update_dict(data + data_fixed, ngram_map)

    def _update_dict(self, data, ngrams):
        for d in data:
            vocab_rev = {val: key for key, val in d['vocab'].iteritems()}
            ngram_map = {"~" + "#".join(vocab_rev.get(i, "$$" + str(i)) for i in key): val for key, val in ngrams.iteritems()}
            d['vocab'].update(ngram_map)

    def _load_data(self, dataset):
        data = []
        for data_file in dataset:
            with open(data_file) as f_in:
                data.append(json.load(f_in))
        return data

    def _get_ngrams(self, data):
        res = []
        for order, order_cnt in zip(self.ngram_order, self.ngram_cnt):
            ngrams = Counter()
            for d in data:
                for seq in d['sequences']:
                    curr_ngram = deque([0] * order, maxlen=order)
                    for w in seq['data']:
                        if type(w) is list:
                            assert len(w) == 1
                            w = w[0]

                        curr_ngram.append(w)
                        ngrams[tuple(curr_ngram)] += 1

                        #print map({val: key for key, val in d['vocab'].iteritems()}.__getitem__, curr_ngram)

            hist = self._build_cummul(ngrams)

            thresh = 0
            for occur, n_ngrams in hist:
                if n_ngrams < order_cnt:
                    thresh = occur
                else:
                    break

            print thresh

            for ngram, ngram_cnt in ngrams.iteritems():
                if ngram_cnt > thresh:
                    res.append(ngram)

        return set(res)

    def _add_ngrams(self, data, ngram_map):
        max_ngram_order = max(len(x) for x in ngram_map)

        for d in data:
            print 'processing data'
            for seq in d['sequences']:
                new_seq = []
                new_seq_score = []
                #print '----------------------------------------'
                #print seq['data_debug']
                curr_ngram = deque([0] * max_ngram_order, maxlen=max_ngram_order)

                y = 0
                lbl_times = set(lbl['time'] for lbl in seq['labels'])
                for t, (w, score) in enumerate(zip(seq['data'], seq['data_score'])):
                    new_seq.append(w)
                    new_seq_score.append(score)

                    wcn = False
                    if type(w) is list:
                        wcn = True
                        assert len(w) == 1
                        w = w[0]

                    if t in lbl_times:  # Reset ngram buffer on boundaries of input.
                        curr_ngram = deque([0] * max_ngram_order, maxlen=max_ngram_order)


                    curr_ngram.append(w)

                    #print map({val: key for key, val in d['vocab'].iteritems()}.__getitem__, curr_ngram)

                    for ngram_order in xrange(2, max_ngram_order + 1):
                        ngram_tuple = tuple(curr_ngram)[-ngram_order:]
                        #if 295 in ngram_tuple and 129 in ngram_tuple:
                        #    import ipdb; ipdb.set_trace()

                        if ngram_tuple in ngram_map:
                            new_item = ngram_map[ngram_tuple]

                            if wcn:
                                new_item = [new_item]
                            new_seq.insert(y + 1, new_item)
                            y += 1
                            for label in seq['labels']:
                                if label['time'] >= y:
                                    label['time'] += 1

                    y += 1

                seq['data'] = new_seq
                seq['data_score'] = new_seq_score


    def _build_cummul(self, ngrams):
        cntr = Counter(ngrams.values())
        res = []
        curr_sum = 0
        for cnt, num_ngrams in sorted(cntr.items(), reverse=True):
            curr_sum += num_ngrams
            res.append((cnt, curr_sum))
        return res





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

    def _save_data(self, data, dataset, outdir):
        for d, dataset_fn in zip(data, dataset):
            out_dataset_fn = os.path.join(outdir, os.path.basename(dataset_fn))
            print 'writing', out_dataset_fn
            with open(out_dataset_fn, 'w') as f_out:
                json.dump(d, f_out, indent=4, sort_keys=True)


if __name__ == '__main__':
    import utils
    utils.init_logging('NGramEnricher')
    utils.pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='append')
    parser.add_argument('--fixed_dataset', action='append', default=[])
    parser.add_argument('--ngram', action='append', required=True)
    parser.add_argument('--outdir', required=True)

    #parser.add_argument('--cutoff_thresh', type=int, required=True)

    args = parser.parse_args()

    main(**vars(args))