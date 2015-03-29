from unittest import TestCase


class TestModel(TestCase):
    def test_how_theano_handles_negative_indicies(self):
        import theano
        import theano.tensor as tt

        i = tt.iscalar()
        x = tt.vector()

        f = theano.function([x, i], x[i])

        self.assertEqual(f([1,2,3,4], -1), 4)
        self.assertEqual(f([1,2,3,4], -2), 3)
        self.assertEqual(f([1,2,3,4], 1), 2)