#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0111

import argparse
from collections import defaultdict
import json
import math
import os.path
import copy
import sys

from collections import namedtuple


ASRHyp = namedtuple('ASRHyp', ['hyp', 'score'])
SLUHyp = namedtuple('SLUHyp', ['acts', 'score'])
DialogAct = namedtuple('DialogAct', ['act', 'slots'])
Slot = namedtuple('Slot', ['name', 'value'])

NEAR_INF = sys.float_info.max


class Dialog(object):
    """Dialog log.

    Representation of one dialog.

    Attributes:
        turns: A list of dialog turns.
        session_id: ID of the dialog.
    """

    def __init__(self, log, labels):
        """Initialises a dialogue object from the external format.

        Keyword arguments:
            log: the object captured as JSON in the `log' file
            labels: the object captured as JSON in the `labels' file
            regress_to_dais: whether to regress DA scores to scores of single
                             DAIs
            norm_slu_scores: whether scores for SLU hypotheses should be
                             normalised to the scale [0, 1]
            slot_normaliser: instance of a normaliser with normalise method

        """
        self.turns = []
        self.session_id = log['session-id']

        if labels:
            for turn_json, turn_label in zip(log['turns'], labels['turns']):
                self.turns.append(Turn(turn_json, turn_label))
        else:
            for turn_json in log['turns']:
                self.turns.append(Turn(turn_json, None))

    def pretty_print(self, indent=0, step=2):
        repr_str = indent * ' ' + 'Dialog:\n'
        repr_str += (indent + step) * ' ' + 'id: "%s",\n' % self.session_id
        repr_str += (indent + step) * ' ' + 'turns:\n'
        for turn in self.turns:
            repr_str += turn.pretty_print(indent + 2 * step, step) + '\n'
        return repr_str

    def __str__(self):
        return self.pretty_print()

    def __repr__(self):
        return 'Dialog(id="%s")' % self.session_id


class Turn(object):
    """One turn of a dialog.

    Representation of one turn in a dialog. Contains information about
    things the user asked as well as the reply from dialog manager.

    Attributes:
        turn_index: Index of the turn in the dialog.
        transcription: Correct transcription of input.
        input: Input from the user.
        ouput: Output from the dialog manager.
        restart: Whether the dialog manager decided to restart the dialog.
    """

    def __init__(self, turn, labels):
        """Initialises a turn object from the external format.

        Keyword arguments:
            log: the object captured as JSON in the `log' file
            labels: the object captured as JSON in the `labels' file
            regress_to_dais: whether to regress DA scores to scores of single
                             DAIs
            norm_slu_scores: whether scores for SLU hypotheses should be
                             normalised to the scale [0, 1]
            slot_normaliser: instance of a normaliser with normalise method

        """
        self.turn_index = turn['turn-index']
        self.transcription = ''

        if labels is not None:
            self.transcription = labels.get('transcription', None)
            self.input = Input(
                turn['input'],
                labels['goal-labels'],
                labels['requested-slots'],
                labels['method-label']
            )
        else:
            self.input = Input(turn['input'], None, None)

        self.output = Output(turn['output'])

    def pretty_print(self, indent=0, step=2):
        repr_str = indent * ' ' + 'Turn #%d:\n' % self.turn_index
        repr_str += ((indent + step) * ' ' + 'transcription: {0!s}\n'.format(
                     self.transcription))
        repr_str += self.input.pretty_print(indent + step, step)
        repr_str += self.output.pretty_print(indent + step, step)
        repr_str += (indent + step) * ' ' + '\n'
        return repr_str

    def __str__(self):
        return self.pretty_print()


class Input(object):
    """Input from the user.

    Representation of the information dialog manager has about what the
    user said. Contains asr and slu hypotheses.

    Attributes:
        live_asr: A list of asr hypothesis from live system.
        live_slu: A list of slu hypothesis from live system.
        batch_asr: A list of asr hypothesis from batch processing.
        batch_slu: A list of slu hypothesis from batch processing.
    """

    def __init__(self, input_json, user_goal, requested_slots, method):
        """Initialises an input object from the external format.

        Keyword arguments:
            input_json: the object captured as JSON in the `log' file
            labels: the object captured as JSON in the `labels' file
            regress_to_dais: whether to regress DA scores to scores of single
                             DAIs
            norm_slu_scores: whether scores for SLU hypotheses should be
                             normalised to the scale [0, 1]
            slot_normaliser: instance of a normaliser with normalise method

        """
        self.live_asr = []
        self.live_slu = []
        self.batch_asr = []
        self.batch_slu = []

        self.user_goal = user_goal
        self.requested_slots = requested_slots
        self.method = method

        for fldname, asr_field, slu_field in (
                ('live', self.live_asr, self.live_slu),
                # ('batch', self.batch_asr, self.batch_slu)
            ):
            if fldname in input_json:
                for asr_hyp in input_json[fldname]['asr-hyps']:
                    asr_field.append(ASRHyp(hyp=asr_hyp['asr-hyp'],
                                            score=asr_hyp['score']))

                slu_hyps = input_json[fldname]['slu-hyps']

                slu_scores = [hyp['score'] for hyp in slu_hyps]

                for hyp_idx, slu_hyp in enumerate(slu_hyps):
                    dialog_acts = []
                    score = slu_scores[hyp_idx]

                    for dialog_act in slu_hyp['slu-hyp']:
                        act = dialog_act['act']
                        slots = []
                        slots_dict = {}
                        da_slots = set()

                        for slot in dialog_act['slots']:
                            slot_name = slot[0]
                            slot_value = str(slot[1]).lower()

                            da_slots.add(slot_name)

                            slots.append(Slot(name=slot_name,
                                              value=slot_value))
                            slots_dict[slot_name] = slot_value

                        dialog_acts.append(DialogAct(act=act,
                                                     slots=tuple(slots)))

                    slu_field.append(SLUHyp(score=score,
                                            acts=dialog_acts))

    @property
    def all_slu(self):
        return self.live_slu + self.batch_slu

    @property
    def all_asr(self):
        return self.live_asr + self.batch_asr

    @property
    def all_slots(self):
        slots = []
        for slu_hyp in self.all_slu:
            for da in slu_hyp.acts:
                if da.act == 'inform':
                    slots.extend(da.slots)
        return slots

    def pretty_print(self, indent=0, step=2):
        repr_str = indent * ' ' + 'Input:\n'
        repr_str += (indent + step) * ' ' + 'Live ASR:\n'
        for asr_hyp in self.live_asr:
            repr_str += (indent + 2 * step) * ' ' + repr(asr_hyp) + '\n'

        repr_str += (indent + step) * ' ' + 'Live SLU:\n'
        for slu_hyp in self.live_slu:
            repr_str += (indent + 2 * step) * ' ' + repr(slu_hyp) + '\n'

        repr_str += (indent + step) * ' ' + 'Batch ASR:\n'
        for asr_hyp in self.batch_asr:
            repr_str += (indent + 2 * step) * ' ' + repr(asr_hyp) + '\n'

        repr_str += (indent + step) * ' ' + 'Batch SLU:\n'
        for slu_hyp in self.batch_slu:
            repr_str += (indent + 2 * step) * ' ' + repr(slu_hyp) + '\n'

        return repr_str

    def str(self):
        return self.pretty_print()

    def DiscreteFact__(self):
        return self.pretty_print()


class Output(object):
    """Input for the dialog manager.

    Attributes:
        transcript: Transcript of the output.
        dialog_acts: A list of dialog acts.
    """

    def __init__(self, output_json):
        if 'dialog-acts' in output_json:
            self.transcript = output_json['transcript']
            self.dialog_acts = []
            for act in output_json['dialog-acts']:
                slots = []
                for slot in act['slots']:
                    # coerce the value to a string and lowercase it
                    slot[1] = str(slot[1]).lower()

                    slots.append(Slot(name=slot[0],
                                      value=slot[1]))
                self.dialog_acts.append(DialogAct(act=act['act'],
                                                  slots=slots))
        else:
            self.transcript = ''
            self.dialog_acts = []

    def pretty_print(self, indent=0, step=2):
        repr_str = indent * ' ' + 'Output:\n'

        repr_str += (indent + step) * ' ' + 'Transcript:\n'
        repr_str += (indent + 2 * step) * ' ' + self.transcript + '\n'

        repr_str += (indent + step) * ' ' + 'Acts:\n'
        for act in self.dialog_acts:
            repr_str += (indent + 2 * step) * ' ' + repr(act) + '\n'

        return repr_str

    def __str__(self):
        return self.pretty_print()


def parse_dialog_from_directory(dialog_dir):
    """
    Keyword arguments:
        dialog_dir: the directory immediately containing the dialogue JSON logs
        regress_to_dais: whether to regress DA scores to scores of single DAIs
        norm_slu_scores: whether scores for SLU hypotheses should be
                         normalised to the scale [0, 1]
        slot_normaliser: instance of a normaliser with normalise method
        reranker_model: if given, an SLU reranker will be applied, using the
                        trained model whose file name is passed in this
                        argument

    """
    log = json.load(open(os.path.join(dialog_dir, 'log.json')))

    labels_file_name = os.path.join(dialog_dir, 'label.json')
    if os.path.exists(labels_file_name):
        labels = json.load(open(labels_file_name))
    else:
        labels = None

    d = Dialog(log, labels)

    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load dstc data.")
    parser.add_argument('-d',
                        required=True,
                        nargs="+",
                        dest='dirs',
                        metavar="dir",
                        help="Directories with logs.")
    args = parser.parse_args()

    dialogs = []
    for directory in args.dirs:
        dialogs.append(
            parse_dialog_from_directory(directory))
    import ipdb; ipdb.set_trace()
