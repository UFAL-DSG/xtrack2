from unittest import TestCase
import mox
from StringIO import StringIO
import numpy as np

from xtrack_data2 import *

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