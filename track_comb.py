import json
import copy


def main(gold_tracker, our_tracker, out_tracker):
    gt = json.load(open(gold_tracker))
    ot = json.load(open(our_tracker))

    for gsess, osess in zip(gt['sessions'], ot['sessions']):
        assert gsess['session-id'] == osess['session-id']

        for gturn, oturn in zip(gsess['turns'], osess['turns']):
            for slot in ['food', 'area', 'name', 'pricerange']:
                gval = get_best_hyp(gturn['goal-labels'].get(slot, {}))

                #if slot == 'food' and gval in ['dontcare'] and False:
                if gval in ['dontcare']:
                    if gval:
                        oturn['goal-labels'][slot] = {gval: 1.0}
                    else:
                        oturn['goal-labels'][slot] = {}

    json.dump(ot, open(out_tracker, 'w'), indent=4)


def get_best_hyp(res):
    if len(res) == 0:
        return None

    total_p = sum(res.values())
    null_p = max(0, 1.0 - total_p)

    max_v, max_p = sorted(res.items(), key=lambda x: x[1])[-1]

    if max_p > null_p:
        return max_v
    else:
        return None





if __name__ == '__main__':
    import utils
    utils.pdb_on_error()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('gold_tracker')
    parser.add_argument('our_tracker')
    parser.add_argument('out_tracker')

    args = parser.parse_args()

    main(**vars(args))