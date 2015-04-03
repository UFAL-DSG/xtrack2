from collections import defaultdict, Counter
import json
import os

import data
import xtrack2_config


def diff_state(last, curr):
    diff = {}
    for key in set(last.keys()).union(curr.keys()):
        if last.get(key) != curr.get(key):
            diff[key] = curr.get(key)

            if not diff[key]:
                del diff[key]


    return diff

def main():
    ontology_path = os.path.join(xtrack2_config.data_directory,
                                 'dstc2/scripts/config/ontology_dstc2.json')

    with open(ontology_path) as f_in:
            dstc_ontology = json.load(f_in)



    dialogs = data.load_dialogs('data/xtrack/e2_tagged/train')

    pattern_counts = Counter()

    for dialog in dialogs:
        users_messages = [(m, s) for m, s, a in
                          zip(dialog.messages, dialog.states, dialog.actors)
                          if a == 1]
        last_state = {}
        for msgs, state in users_messages:
            occurence = defaultdict(list)
            for nbest_pos, (msg, msg_score) in enumerate(msgs):
                for key, val in state.iteritems():
                    if val and val in msg:
                        occurence[key].append(nbest_pos)

            for key, pattern in occurence.iteritems():
                if pattern[0] == 0:
                    pattern_counts[tuple(pattern)] += 1

            last_state = state

    cnt_not_in_1best = 0
    total_cnt = 0
    for pattern, cnt in pattern_counts.most_common():
        value_is_contained_in_nbest = pattern != (0, )
        value_is_not_on_1best = not 1 in pattern
        if value_is_contained_in_nbest:
            if value_is_not_on_1best:
                cnt_not_in_1best += cnt
                print pattern, cnt
            total_cnt += cnt

    print 'not in 1best: ', cnt_not_in_1best * 1.0 / total_cnt




if __name__ == '__main__':
    import utils
    utils.pdb_on_error()
    #import argparse
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()
    #main(**vars(args))
    main()