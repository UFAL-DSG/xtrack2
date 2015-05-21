import json
import os
import dstc_util
import numpy as np

def load_ontology():
    ontology_path = os.path.join('/xdisk/devel/proj/xtrack2/data/dstc2/scripts/config/ontology_dstc2.json')

    with open(ontology_path) as f_in:
        dstc_ontology = json.load(f_in)
        ontology = dict(
            food=dstc_ontology['informable']['food'],
            pricerange=dstc_ontology['informable']['pricerange'],
            area=dstc_ontology['informable']['area'],
            name=dstc_ontology['informable']['name'],
            method=dstc_ontology['method']
        )

    return ontology


def main():
    flist = '/xdisk/devel/proj/xtrack2/data/dstc2/scripts/config/dstc2_dev.flist'
    data_dir = '/xdisk/devel/proj/xtrack2/data/dstc2/data'

    ontology = load_ontology()

    dialog_dirs = []

    with open(flist) as f_in:
        for f_name in f_in:
            dialog_dirs.append(os.path.join(data_dir, f_name.strip()))

    scores = []
    scores_good = []
    scores_hyp_len = []

    cntr = 0
    cntr_nbest = 0
    cntr_nbest_true = 0
    cntr_1best_true = 0
    cntr_1best_no_food = 0
    cntr_1best_no_food_but_nbest_true = 0
    cntr_1best_true_none = 0
    cntr_food = 0
    cntr_turns = 0
    for i, dialog_dir in enumerate(dialog_dirs):
        d = dstc_util.parse_dialog_from_directory(dialog_dir)

        for turn in d.turns:
            cntr_turns += 1
            true_food = None
            for food in ontology['food']:
                if food in turn.transcription:
                    true_food = food

            if true_food:
                cntr_food += 1

                true_food_in_nbest = any(true_food in h.hyp for h in turn.input.live_asr)
                if true_food_in_nbest:
                    cntr_nbest_true += 1

                incorrect_food_in_1best = False
                no_food_in_1best = True
                for food in ontology['food']:
                    food_in_1best = food in turn.input.live_asr[0].hyp
                    this_is_correct_food = food == true_food

                    if food_in_1best:
                        no_food_in_1best = False

                    if food_in_1best and not this_is_correct_food:
                        incorrect_food_in_1best = True

                true_food_in_1best = true_food in turn.input.live_asr[0].hyp

                if true_food_in_1best:
                    cntr_1best_true += 1

                if incorrect_food_in_1best:
                    cntr += 1
                    scores.append(turn.input.live_asr[0].score)
                    scores_hyp_len.append(turn.input.live_asr[0].hyp.split().__len__())
                    if true_food_in_nbest:
                        cntr_nbest += 1

                        #break

                elif not incorrect_food_in_1best and not no_food_in_1best:
                    scores_good.append(turn.input.live_asr[0].score)

                if no_food_in_1best:
                    cntr_1best_no_food += 1
                    if true_food_in_nbest:
                        cntr_1best_no_food_but_nbest_true += 1
            else:
                for food in ontology['food']:
                    food_in_1best = food in turn.input.live_asr[0].hyp

                    if food_in_1best:
                        cntr_1best_true_none += 1
                        scores.append(turn.input.live_asr[0].score)
                        scores_hyp_len.append(turn.input.live_asr[0].hyp.split().__len__())
                        break



    print "#turns                                        (%d)" % cntr_turns
    print "#true_food=none&food_in_1best                 (%d)" % cntr_1best_true_none
    print "#true_food                                    (%d)" % cntr_food
    print "#true_food&true_food_in_1best                 (%d)" % cntr_1best_true
    print "#true_food&no_food_in_1best                   (%d)" % cntr_1best_no_food
    print "#true_food&incorrect_1best                    (%d)" % cntr
    print "#true_food&incorrect_1best&correct_in_nbest   (%d)" % cntr_nbest
    print "#true_food&no_food_1best&correct_in_nbest     (%d)" % cntr_1best_no_food_but_nbest_true
    print "#true_food&true_food_in_nbest                 (%d)" % cntr_nbest_true
    #print '%.2f' % (cntr * 1.0 / cntr_turns), cntr, cntr_turns, , cntr_nbest

    print
    import ipdb; ipdb.set_trace()

    res_scores = []
    for score, hyp_len in zip(scores, scores_hyp_len):
        #score = np.power(np.exp(score), 1.0 / hyp_len)
        score = np.exp(score)
        res_scores.append(score)
    print "wrong 1best mean score: %.2f" % np.mean(res_scores)
    print len(res_scores)

    res_scores = []
    for score in scores_good:
        score = np.exp(score)
        res_scores.append(score)
    print "good 1best mean score: %.2f" % np.mean(res_scores)
    print len(res_scores)



if __name__ == '__main__':
    import utils
    utils.pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(**vars(args))