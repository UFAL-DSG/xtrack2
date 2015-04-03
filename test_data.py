from unittest import TestCase
import mox
from StringIO import StringIO
import numpy as np

from data import *
from data_model import Dialog


class TestDataBuilder(TestCase):
    def setUp(self):
        self.mox = mox.Mox()

        self.slots = ["food", "location", "method", "req_phone"]
        self.slot_groups = {
            "goals": ["food", "location"],
            "method": ["method"],
            "requested": ["req_phone"]
        }
        self.ontology = {
            'method': ['m1', 'm2', 'm3'],
            'food': ['chinese', 'czech', 'russian'],
            'location': ['north', 'west', 'south']
        }



    def _create_test_dialog(self, id, food1):
        d = Dialog(id, id)
        d.add_message([("hello", 0.0)], None, Dialog.ACTOR_SYSTEM)
        d.add_message([("I want %s food" % food1, 0.0)],
                      {'food': food1},
                      Dialog.ACTOR_USER)
        d.add_message([("ok", 0.0)], {'food': food1}, Dialog.ACTOR_SYSTEM)
        d.add_message([("no I want czech", 0.0)],
                      {'food': 'czech'},
                      Dialog.ACTOR_USER)

        return d

    def _create_builder(self, **override_args):
        arg_dict = dict(slots=self.slots,
                        slot_groups=self.slot_groups,
                        based_on=None,
                        include_base_seqs=False,
                        oov_ins_p=0.0,
                        word_drop_p=0.0,
                        include_system_utterances=True,
                        nth_best=0,
                        score_bins=[0.0, 0.5, 0.8],
                        debug_dir=None,
                        tagged=True,
                        ontology=self.ontology,
                        no_label_weight=True)
        if override_args:
            arg_dict.update(override_args)

        builder = DataBuilder(**arg_dict)

        return builder

    def test__create_seq(self):
        d = Dialog("oid", "sessid")
        builder = self._create_builder()

        seq = builder._create_seq(d)

        self.assertEqual(seq.id, 'sessid')
        self.assertEqual(seq.source_dir, 'oid')
        self.assertTrue(type(seq.data) is list)
        self.assertTrue(type(seq.data_score) is list)
        self.assertTrue(type(seq.labels) is list)

    def test__create_new_data_instance(self):
        builder = self._create_builder()

        builder._create_new_data_instance()

        self.assertTrue(builder.xd is not None)

    def test__process_dialog(self):
        self._test__process_dialog(with_system_utterances=True)
        self._test__process_dialog(with_system_utterances=False)

    def _test__process_dialog(self, with_system_utterances):
        if with_system_utterances:
            builder = self._create_builder(nth_best=1)
        else:
            builder = self._create_builder(nth_best=1,
                                       include_system_utterances=False)

        self.mox.StubOutWithMock(builder, '_process_msg')

        d = Dialog("id", "id")
        d.add_message([("hello", 0.0)], None, Dialog.ACTOR_SYSTEM)
        d.add_message([
                          ("chinese food", 0.0),
                          ("good chinese food", 0.3)
                      ],
                      {'food': 'chinese'},
                      Dialog.ACTOR_USER)
        d.add_message([("ok", 0.0)], {'food': 'chinese'}, Dialog.ACTOR_SYSTEM)

        seq = builder._create_seq(d)
        last_state = None
        if with_system_utterances:
            builder._process_msg('hello', 0.0, None, None, Dialog.ACTOR_SYSTEM,
                                 seq, 'hello')
        builder._process_msg('good chinese food', 0.3, {'food': 'chinese'},
                             None, 1, seq, 'chinese food')
        if with_system_utterances:
            builder._process_msg('ok', 0.0, {'food': 'chinese'},
                                 {'food': 'chinese'}, Dialog.ACTOR_SYSTEM,
                                 seq, 'ok')
        self.mox.ReplayAll()

        builder._process_dialog(d, seq)

        self.mox.VerifyAll()

    def test__process_msg(self):
        self._test__process_msg(True)
        self._test__process_msg(False)

    def _test__process_msg(self, tagged):
        builder = self._create_builder(nth_best=1, tagged=tagged)
        builder._create_new_data_instance()
        seq = builder._create_seq(Dialog("id", "id"))

        builder._process_msg('I want @chinese food', 0.9,
                             {'food': 'chinese'},
                             None, Dialog.ACTOR_USER, seq, "chinese only")
        builder._process_msg('czech food', 0.3,
                             {'food': 'czech'},
                             None, Dialog.ACTOR_USER, seq, "czech")

        if not tagged:
            self.assertItemsEqual(seq.data, [builder.xd.vocab[w] for w  in
                                             ['i', 'want', 'chinese', 'food',
                                              'czech', 'food']])
            self.assertEqual(
                seq.labels[0]['slots']['food'],
                builder.xd.classes['food']['chinese']
            )
            self.assertEqual(
                seq.labels[1]['slots']['food'],
                builder.xd.classes['food']['czech']
            )
        else:

            self.assertItemsEqual(seq.data, [builder.xd.vocab[w] for w  in
                                             ['i', 'want', '#food0#', 'food',
                                              '#food1#', 'food']])
            self.assertEqual(
                seq.labels[0]['slots']['food'],
                builder.xd.classes['food']['#food0']
            )
            self.assertEqual(
                seq.labels[1]['slots']['food'],
                builder.xd.classes['food']['#food1']
            )


    def test_build(self):
        builder = self._create_builder()

        dialogs = []
        d = Dialog("1", "1")
        d.add_message([("hello", 0.0)], None, Dialog.ACTOR_SYSTEM)
        d.add_message([("I want chinese food", 0.0)],
                      {'food': 'chinese'},
                      Dialog.ACTOR_USER)
        d.add_message([("ok", 0.0)], {'food': 'chinese'}, Dialog.ACTOR_SYSTEM)
        d.add_message([("no I want czech", 0.0)],
                      {'food': 'czech'},
                      Dialog.ACTOR_USER)
        dialogs.append(d)

        d = Dialog("2", "2")
        d.add_message([("I want russian food", 0.0)],
                      {'food': 'russian'},
                      Dialog.ACTOR_USER)
        dialogs.append(d)

        seq1 = builder._create_seq(dialogs[0])
        seq2 = builder._create_seq(dialogs[1])

        self.mox.StubOutWithMock(builder, '_create_seq')
        self.mox.StubOutWithMock(builder, '_process_dialog')
        self.mox.StubOutWithMock(builder, '_perform_sanity_checks')
        self.mox.StubOutWithMock(builder, '_append_seq_if_nonempty')
        builder._create_seq(dialogs[0]).AndReturn(seq1)
        builder._process_dialog(dialogs[0], seq1)
        builder._perform_sanity_checks(seq1)
        builder._append_seq_if_nonempty(seq1)

        builder._create_seq(dialogs[1]).AndReturn(seq2)
        builder._process_dialog(dialogs[1], seq2)
        builder._perform_sanity_checks(seq2)
        builder._append_seq_if_nonempty(seq2)
        self.mox.ReplayAll()

        xd = builder.build(dialogs)
        xd.save('/dev/null')

        self.mox.VerifyAll()

