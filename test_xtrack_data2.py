from unittest import TestCase
import mox
from StringIO import StringIO
import numpy as np

from xtrack_data2 import *
from data_model import Dialog


class TestXTrackData2(TestCase):
    def test_functions(self):
        tokens = list(tokenize("this IS, my Text!ok?"))
        self.assertItemsEqual(['this', 'IS', 'my', 'Text', 'ok'], tokens)

    def setUp(self):
        self.m = mox.Mox()
        self.xd = XTrackData2()

    def test__init(self):
        self.m.StubOutWithMock(self.xd, '_init_after_load')
        self.xd._init_after_load()
        self.m.ReplayAll()

        slots = ['food', 'area']
        self.xd._init(slots, {'goals': ['food', 'area']}, None,
                      False)

        self.m.UnsetStubs()
        self.m.VerifyAll()

        self.assertTrue(self.xd.vocab['#NOTHING'] == 0)
        self.assertTrue('#EOS' in self.xd.vocab)
        self.assertTrue('#OOV' in self.xd.vocab)

        for slot in slots:
            self.assertTrue(slot in self.xd.classes)


    def test__init_after_load(self):
        self.fail()

    def test__process_msg(self):
        slots = ['food', 'area']
        self.xd._init(slots, {'goals': ['food', 'area']}, None,
                      False)

        seq = {
            'data': [],
            'data_score': [],
            'data_actor': [],
            'data_switch': [],
            'data_debug': [],
            'labels': [],
            'true_input': []
        }
        f_dump_text = f_dump_cca = StringIO()
        self.xd._process_msg('I want chinese food', np.log(0.5), {'food':
                                                                      'chinese'},
                             {'food': None}, data_model.Dialog.ACTOR_USER,
                             seq, 0.0, 0.0, None, f_dump_text, f_dump_cca,
                             'I want chinese food', [0.3, 0.6, 1.0])

        self.assertEqual(seq['data'], [3, 4, 5, 6])
        self.assertEqual(seq['data_score'], [1, 1, 1, 1])
        self.assertEqual(seq['data_actor'], [1, 1, 1, 1])

        self.assertEqual(len(seq['labels']), 1)
        self.assertEqual(seq['labels'][0]['time'], 3)
        self.assertEqual(seq['labels'][0]['slots']['food'], 1)


    def test__sample_paths(self):
        self.fail()

    def test__split_dialog(self):
        self.fail()

    def test_build(self):
        self.fail()

    def test__build_token_features(self):
        self.fail()

    def test__compute_stats(self):
        self.fail()

    def test__normalize(self):
        self.fail()

    def test_get_token_ndx(self):
        self.fail()

    def test_state_to_label(self):
        self.fail()

    def test_state_to_label_for(self):
        self.fail()


class TestXTrackData2Builder(TestCase):
    def setUp(self):
        self.slots = ["food", "location", "method", "req_phone"]
        self.slot_groups = {"goals": ["food", "location"], "method": [
            "method"], "requested": ["req_phone"]}
        self.ontology = {
            'method': ['m1', 'm2', 'm3'],
            'food': ['chinese', 'czech', 'russian'],
            'location': ['north', 'west', 'south']
        }

        d = Dialog("1", "1")
        d.add_message([("hello", 0.0)], None, Dialog.ACTOR_SYSTEM)
        d.add_message([("I want chinese food", 0.0)],
                      {'food': 'chinese'},
                      Dialog.ACTOR_USER)
        d.add_message([("ok", 0.0)], {'food': 'chinese'}, Dialog.ACTOR_SYSTEM)
        d.add_message([("no I want czech", 0.0)],
                      {'food': 'czech'},
                      Dialog.ACTOR_USER)

        self.dialogs = [
            d
        ]

    def _create_builder(self):
        builder = XTrackData2Builder(slots=self.slots,
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
                                     no_label_weight=True
        )

        return builder

    def test__create_seq(self):
        d = Dialog("oid", "sessid")
        builder = self._create_builder()

        seq = builder._create_seq(d)

        self.assertEqual(seq['id'], 'sessid')
        self.assertEqual(seq['source_dir'], 'oid')
        self.assertTrue('data' in seq)
        self.assertTrue('data_score' in seq)
        self.assertTrue('labels' in seq)

    def test_build(self):
        builder = self._create_builder()

        xd = builder.build(self.dialogs)

        print xd.sequences[0]
        print xd.sequences[0]['tags']
