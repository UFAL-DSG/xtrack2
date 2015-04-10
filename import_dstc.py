import argparse
import os
import random

import dstc_util
from data_model import Dialog


ontology = {
    "method": [
        "byconstraints",
        "byname",
        "finished",
        "byalternatives"
    ],
    "food": [
            "afghan",
            "african",
            "afternoon tea",
            "asian oriental",
            "australasian",
            "australian",
            "austrian",
            "barbeque",
            "basque",
            "belgian",
            "bistro",
            "brazilian",
            "british",
            "canapes",
            "cantonese",
            "caribbean",
            "catalan",
            "chinese",
            "christmas",
            "corsica",
            "creative",
            "crossover",
            "cuban",
            "danish",
            "eastern european",
            "english",
            "eritrean",
            "european",
            "french",
            "fusion",
            "gastropub",
            "german",
            "greek",
            "halal",
            "hungarian",
            "indian",
            "indonesian",
            "international",
            "irish",
            "italian",
            "jamaican",
            "japanese",
            "korean",
            "kosher",
            "latin american",
            "lebanese",
            "light bites",
            "malaysian",
            "mediterranean",
            "mexican",
            "middle eastern",
            "modern american",
            "modern eclectic",
            "modern european",
            "modern global",
            "molecular gastronomy",
            "moroccan",
            "new zealand",
            "north african",
            "north american",
            "north indian",
            "northern european",
            "panasian",
            "persian",
            "polish",
            "polynesian",
            "portuguese",
            "romanian",
            "russian",
            "scandinavian",
            "scottish",
            "seafood",
            "singaporean",
            "south african",
            "south indian",
            "spanish",
            "sri lankan",
            "steakhouse",
            "swedish",
            "swiss",
            "thai",
            "the americas",
            "traditional",
            "turkish",
            "tuscan",
            "unusual",
            "vegetarian",
            "venetian",
            "vietnamese",
            "welsh",
            "world"
        ],
        "pricerange": [
            "cheap",
            "moderate",
            "expensive"
        ],
        "name": [
            "ali baba",
            "anatolia",
            "ask",
            "backstreet bistro",
            "bangkok city",
            "bedouin",
            "bloomsbury restaurant",
            "caffe uno",
            "cambridge lodge restaurant",
            "charlie chan",
            "chiquito restaurant bar",
            "city stop restaurant",
            "clowns cafe",
            "cocum",
            "cote",
            "cotto",
            "curry garden",
            "curry king",
            "curry prince",
            "curry queen",
            "da vinci pizzeria",
            "da vince pizzeria",
            "darrys cookhouse and wine shop",
            "de luca cucina and bar",
            "dojo noodle bar",
            "don pasquale pizzeria",
            "efes restaurant",
            "eraina",
            "fitzbillies restaurant",
            "frankie and bennys",
            "galleria",
            "golden house",
            "golden wok",
            "gourmet burger kitchen",
            "graffiti",
            "grafton hotel restaurant",
            "hakka",
            "hk fusion",
            "hotel du vin and bistro",
            "india house",
            "j restaurant",
            "jinling noodle bar",
            "kohinoor",
            "kymmoy",
            "la margherita",
            "la mimosa",
            "la raza",
            "la tasca",
            "lan hong house",
            "little seoul",
            "loch fyne",
            "mahal of cambridge",
            "maharajah tandoori restaurant",
            "meghna",
            "meze bar restaurant",
            "michaelhouse cafe",
            "midsummer house restaurant",
            "nandos",
            "nandos city centre",
            "panahar",
            "peking restaurant",
            "pipasha restaurant",
            "pizza express",
            "pizza express fen ditton",
            "pizza hut",
            "pizza hut city centre",
            "pizza hut cherry hinton",
            "pizza hut fen ditton",
            "prezzo",
            "rajmahal",
            "restaurant alimentum",
            "restaurant one seven",
            "restaurant two two",
            "rice boat",
            "rice house",
            "riverside brasserie",
            "royal spice",
            "royal standard",
            "saffron brasserie",
            "saigon city",
            "saint johns chop house",
            "sala thong",
            "sesame restaurant and bar",
            "shanghai family restaurant",
            "shiraz restaurant",
            "sitar tandoori",
            "stazione restaurant and coffee bar",
            "taj tandoori",
            "tandoori palace",
            "tang chinese",
            "thanh binh",
            "the cambridge chop house",
            "the copper kettle",
            "the cow pizza kitchen and bar",
            "the gandhi",
            "the gardenia",
            "the golden curry",
            "the good luck chinese food takeaway",
            "the hotpot",
            "the lucky star",
            "the missing sock",
            "the nirala",
            "the oak bistro",
            "the river bar steakhouse and grill",
            "the slug and lettuce",
            "the varsity restaurant",
            "travellers rest",
            "ugly duckling",
            "venue",
            "wagamama",
            "yippee noodle bar",
            "yu garden",
            "zizzi cambridge"
        ],
        "area": [
            "centre",
            "north",
            "west",
            "south",
            "east"
        ]
}

def _stringify_act(acts):
    res = []
    for act in acts:
        if len(act.slots) > 0:
            for slot_name, slot_value in act.slots:
                res.append(act.act)
                res.append(slot_name)
                res.append(slot_value.replace(' ', '_'))
        else:
            res.append("%s" % act.act)

    if len(res) == 0:
        res = ["sys"]
    return " ".join(res)


def import_dstc(data_dir, out_dir, flist, constraint_slots,
                requestable_slots,
                use_stringified_system_acts):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    requestable_slots = requestable_slots.split(',')

    dialog_dirs = []
    #for root, dirs, files in os.walk(data_dir, followlinks=True):
    #    for f_name in files:
    #        if f_name == 'log.json':
    #            dialog_dirs.append(root)
    with open(flist) as f_in:
        for f_name in f_in:
            dialog_dirs.append(os.path.join(data_dir, f_name.strip()))

    for i, dialog_dir in enumerate(dialog_dirs):
        dialog = dstc_util.parse_dialog_from_directory(dialog_dir)

        out_dialog = Dialog(dialog_dir, dialog.session_id)
        last_state = None
        for turn in dialog.turns:
            if use_stringified_system_acts:
                msg = _stringify_act(turn.output.dialog_acts)
            else:
                msg = turn.output.transcript
            out_dialog.add_message([(msg, 0.0)],
                                   last_state,
                                   Dialog.ACTOR_SYSTEM)
            state = dict(turn.input.user_goal)

            state['method'] = (turn.input.method if turn.input.method != 'none'
                                                 else None)
            for slot in requestable_slots:
                if slot in turn.input.requested_slots:
                    state['req_%s' % slot] = '@_yes'

            user_messages = [(turn.transcription, 0.0)]
            for hyp in turn.input.live_asr:
                user_messages.append((hyp.hyp, hyp.score))


            out_dialog.add_message(
                user_messages,
                state,
                Dialog.ACTOR_USER
            )

            last_state = state  #turn.input.user_goal

        with open(os.path.join(out_dir, "%d.json" % (i,)
                 ), "w") as f_out:
            f_out.write(out_dialog.serialize())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Import DSTC data to XTrack2.")
    parser.add_argument('--data_dir',
                        required=True,
                        help="Root directory with logs.")
    parser.add_argument('--flist',
                        required=True,
                        help="File list with logs.")
    parser.add_argument('--out_dir',
                        required=True,
                        help="Output directory.")
    parser.add_argument('--constraint_slots', default='food,area,pricerange,'
                                                     'name')
    parser.add_argument('--requestable_slots', default='food,area,pricerange,'
                                                       'name,addr,phone,'
                                                       'postcode,signature')
    parser.add_argument('--use_stringified_system_acts', action='store_true',
                        default=False)

    args = parser.parse_args()

    import_dstc(**vars(args))
