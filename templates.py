from collections import defaultdict, namedtuple
import json
import random

from data_utils import import_dstc_data, load_ontology
from data_model import Dialog


class DialogTemplate(object):
    def __init__(self, slots=None, values=None, msgs=None):
        if slots:
            self.slots = slots
        else:
            self.slots = {}

        if values:
            self.values = values
        else:
            self.values = defaultdict(dict)

        if msgs:
            self.msgs = msgs
        else:
            self.msgs = []

    def add_message(self, tpl, tpl_state, asr_score, actor, slots_mentioned):
        self.msgs.append((tpl, tpl_state, asr_score, actor, slots_mentioned))

    def fill(self, ontology):
        substs = {}
        for slot, slot_vals in ontology.iteritems():
            if slot in self.values:
                for i in range(len(self.values[slot])):
                    substs['#value_%s%d#' % (slot, i)] = (slot, random.choice(slot_vals))

        res = []
        for tpl, tpl_state, asr_score, actor, slots_mentioned in self.msgs:
            state = dict(tpl_state)
            for pattern, (slot, val) in substs.iteritems():
                tpl = tpl.replace(pattern, val)
                if slot in state and state[slot] == pattern:
                    state[slot] = val
            res.append((tpl, state, asr_score, actor, slots_mentioned))

        return res


    def get_slot_token(self, key):
        if not key in self.slots:
            self.slots[key] = len(self.slots)

        return "slot%d" % self.slots[key]

    def get_value_token(self, slot, value):
        if not value in self.values[slot]:
            self.values[slot][value] = len(self.values[slot])

        return "#value_%s%d#" % (slot, self.values[slot][value])

    def serialize(self):
        return {'slots': self.slots, 'values': self.values, 'msgs': self.msgs}

    @staticmethod
    def deserialize(tpl):
        return DialogTemplate(
            slots=tpl['slots'],
            values=tpl['values'],
            msgs=tpl['msgs']
        )


class TemplateExtractor(object):
    def __init__(self, templates=None):
        if templates:
            self.templates = templates
        else:
            self.templates = []

    def add_dialog(self, dialog):
        tpl = DialogTemplate()
        curr_state = {}

        for actor, msgs, state, slots_mentioned in zip(dialog.actors, dialog.messages, dialog.states, dialog.slots_mentioned):
            if actor == dialog.ACTOR_USER:
                for msg, asr_score in msgs[:2]:
                    tpl_str, tpl_state = self._extract_template(tpl, msg, curr_state, state)

                    tpl.add_message(tpl_str, tpl_state, asr_score, actor, slots_mentioned)

        self.templates.append(tpl)

        #ontology = load_ontology(open('data/dstc2/scripts/config/ontology_dstc2.json'))
        #for x in tpl.fill(ontology):
        #    print x
        #
        #import ipdb; ipdb.set_trace()

    def _extract_template(self, tpl, msg, curr_state, new_state):
        if new_state:
            res_state = {}
            for key, val in new_state.iteritems():
                if not key in {'food', 'pricerange', 'area', 'name'}:
                    res_state[key] = val
                else:
                    value_tag = tpl.get_value_token(key, val)

                    if val and val in msg:
                        msg = msg.replace(val, value_tag)
                    else:
                        value_tag = val

                    res_state[key] = value_tag

            curr_state.update(res_state)

        return msg, dict(curr_state)

    def sample_dialogs(self, ontology):
        for i, dialog in enumerate(self.templates):
            res_dialog = Dialog('tpl%d' % i, 'tpl%d' % i)
            for tpl, state, asr_score, actor, slots_mentioned in dialog.fill(ontology):
                res_dialog.add_message([(tpl, asr_score)], None, state, actor, slots_mentioned)

            yield res_dialog

    def serialize(self):
        res = []
        for tpl in self.templates:
            res.append(tpl.serialize())

        return res

    def save(self, file_name):
        with open(file_name, 'w') as f_out:
            json.dump(self.serialize(), f_out, indent=4)

    @staticmethod
    def load(file_name):
        res = TemplateExtractor()

        with open(file_name) as f_in:
            data = json.load(f_in)

            for tpl in data:
                res.templates.append(DialogTemplate.deserialize(tpl))

        return res



def main(data_dir, dataset, out_file):
    dialogs = import_dstc_data(data_directory=data_dir,
                               dataset=dataset)
    tpls = TemplateExtractor()

    for dialog in dialogs:
        tpls.add_dialog(dialog)

    tpls.save(out_file)



if __name__ == '__main__':
    import utils
    utils.pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out_file', required=True)


    args = parser.parse_args()

    main(**vars(args))