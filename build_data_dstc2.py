import json
import os

import data
import data_utils
import xtrack2_config


def main(builder_type, only_slot, tagged, concat_whole_nbest, include_whole_nbest,
         use_wcn, ngrams, split_dialogs, sample_subdialogs, train_nbest_entries,
         e_name):
    import utils
    utils.pdb_on_error()

    ontology_path = os.path.join(xtrack2_config.data_directory,
                                 'dstc2/scripts/config/ontology_dstc2.json')
    with open(ontology_path) as f_in:
        dstc_ontology = json.load(f_in)
        ontology = dict(
            food=dstc_ontology['informable']['food'],
            pricerange=dstc_ontology['informable']['pricerange'],
            area=dstc_ontology['informable']['area'],
            name=dstc_ontology['informable']['name'],
            method=dstc_ontology['method']
        )
        ontology['method'].remove('none')

    slots = ['food', 'area', 'pricerange', 'name', 'method', 'req_food',
             'req_area', 'req_pricerange', 'req_name', 'req_phone',
             'req_addr', 'req_postcode', 'req_signature']

    slot_groups = dict(
        food=['food'],
        area=['area'],
        pricerange=['pricerange'],
        name=['name'],
        method=['method'],
        goals=['food', 'area', 'pricerange', 'name'],
        requested=['req_food', 'req_area', 'req_pricerange', 'req_name',
                   'req_phone', 'req_addr', 'req_postcode', 'req_signature']
    )

    experiment_name = e_name
    if tagged:
        experiment_name += '_tagged'

    if concat_whole_nbest:
        experiment_name += '_nbest'
        nth_best=None
    else:
        if train_nbest_entries:
            nth_best=map(int, train_nbest_entries.split(','))
            experiment_name += '_%sbest' % ("n".join(map(str, nth_best)), )
        else:
            nth_best = [1]
            experiment_name += '_1best'

    if use_wcn:
        experiment_name += '_wcn'
    else:
        experiment_name += '_nowcn'

    if ngrams:
        experiment_name += '_ngrams'

    if split_dialogs:
        experiment_name += '_split'

    experiment_name += '_%s' % builder_type

    if only_slot:
        slots = only_slot.split(',')
        slot_groups = {}
        for slot in slots:
            slot_groups[slot] = [slot]
        slot_groups['all'] = slots
        experiment_name += "_%s" % only_slot.replace(',', '-')

    data_utils.prepare_experiment(
        experiment_name=experiment_name,
        data_directory=xtrack2_config.data_directory,
        slots=slots,
        slot_groups=slot_groups,
        ontology=ontology,
        builder_opts=dict(
            tagged=tagged,
            no_label_weight=True,
            tag_only=['panasian', 'basque', 'jamaican', 'singaporean', 'polish', 'russian', 'venetian', 'creative', 'welsh', 'australasian', 'scottish', 'world', 'malaysian', 'unusual', 'vegetarian', 'indonesian', 'swiss', 'caribbean', 'cantonese', 'danish', 'australian', 'brazilian', 'persian', 'fusion', 'english', 'irish', 'christmas', 'corsica', 'austrian', 'kosher', 'canapes', 'bistro', 'belgian', 'moroccan', 'traditional', 'afghan', 'barbeque', 'romanian', 'german', 'steakhouse', 'greek', 'cuban', 'african', 'scandinavian', 'japanese', 'polynesian', 'seafood', 'eritrean', 'swedish', 'catalan', 'lebanese', 'tuscan', 'mexican']
        ),
        builder_type=builder_type,
        use_wcn=use_wcn,
        ngrams=ngrams,
        concat_whole_nbest=concat_whole_nbest,
        include_whole_nbest=include_whole_nbest,
        split_dialogs=split_dialogs,
        sample_subdialogs=sample_subdialogs,
        nth_best=nth_best
    )

    print experiment_name


if __name__ == '__main__':
    import utils
    utils.init_logging('BuildDataDSTC2')

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--builder_type', default='xtrack')
    parser.add_argument('--only_slot', default=None)
    parser.add_argument('--tagged', action='store_true', default=False)
    parser.add_argument('--ngrams', default=None)
    parser.add_argument('--concat_whole_nbest', action='store_true', default=False)
    parser.add_argument('--include_whole_nbest', action='store_true', default=False)
    parser.add_argument('--use_wcn', action='store_true', default=False)
    parser.add_argument('--split_dialogs', action='store_true', default=False)
    parser.add_argument('--e_name', default='xx')
    parser.add_argument('--sample_subdialogs', type=int, default=0)
    parser.add_argument('--train_nbest_entries', type=str, default="1")

    args = parser.parse_args()
    main(**vars(args))
