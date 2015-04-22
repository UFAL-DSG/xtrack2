import re
import collections

from nltk.metrics.confusionmatrix import ConfusionMatrix

def extract_val(val):
    return re.match(r'.*\((?P<val>[^\)]*)\)', val).group('val')


def parse_ln(ln):
    res = re.match(r'(?P<slot>[a-z_]*).*lbl\((?P<lbl>[^\)]*)\).*pred\((?P<pred>[^\)]*)\)', ln)

    slot = res.group('slot')
    lbl = res.group('lbl')
    pred = res.group('pred')

    return slot, lbl, pred


def main(track_file):
    data = collections.defaultdict(list)
    with open(track_file) as f_in:
        for ln in f_in:
            ln = ln.strip()
            if 'lbl(' in ln:
                try:
                    slot, lbl, pred = parse_ln(ln)
                except:
                    print 'ERR PARSING'
                data[slot].append((lbl, pred))

    c = ConfusionMatrix(*zip(*data['food']))

    print c.pp(show_percents=False, truncate=10, sort_by_count=True)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('track_file')

    args = parser.parse_args()


    main(**vars(args))