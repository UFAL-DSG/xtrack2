import numpy as np
import theano
import theano.tensor as tt

from layers import *
import updates


xor_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

xor_data_y = [
    0,
    1,
    1,
    0
]


def test_unwrapper():
    emb_size = 5
    y_time = tt.ivector()
    y_seq_id = tt.ivector()
    x = tt.tensor3()

    emb = IdentityInput(x, size=5)

    sequn = SeqUnwrapper(20)
    sequn.connect(emb, y_time, y_seq_id)

    rng = np.random.RandomState(23455)
    conv = LeNetConvPoolLayer()
    conv.connect(sequn, rng, (3, 1, 5, emb_size), (1, 1, ))
    #prev_layer = conv

    f = theano.function([x, y_time, y_seq_id], conv.output())

    xx = np.random.randn(20, 4, emb_size)
    y_time = [3, 7, 10, 12]
    y_seq_id = [0, 0, 0, 0]
    res = f(xx, y_time, y_seq_id)
    print res.shape
    print res
    import ipdb; ipdb.set_trace()




def test_mlp():
    y_true = tt.ivector()
    x = tt.imatrix()

    emb = IdentityInput(x, size=2)

    mlp = MLP([2, 16, 2], ['linear', 'tanh', 'softmax'])
    mlp.connect(emb)
    _predict = theano.function([x], mlp.output())
    #print _predict(xor_data)

    objective = CrossEntropyObjective()
    objective.connect(mlp, y_true)

    cost = objective.output()
    params = list(objective.get_params())

    updater = updates.RProp(lr=0.1)
    model_updates = updater.get_updates(params, cost)

    _train = theano.function([x, y_true], cost, updates=model_updates)

    for i in range(100):
        print _train(xor_data, xor_data_y)

    print _predict(xor_data)


def test_cherrypick():
    np_x = np.random.randn(10, 7, 2)
    input_layer = MatrixInput(np_x)

    y_time = tt.ivector()
    y_seq_id = tt.ivector()

    cpt = CherryPick()
    cpt.connect(input_layer, y_time, y_seq_id)

    f = theano.function([y_time, y_seq_id], cpt.output())

    print 'Orig matrix:'
    print np_x
    print f([0, 7, 9], [0, 1, 2])


def test_verify_exprgrad():
    from theano import tensor
    import numpy
    x = tt.scalar()
    f = theano.function([x], x)
    #def f(x):
    #    return x

    x_val = numpy.asarray([0.1, 0.2])

    rng = numpy.random.RandomState(42)

    print 'going'
    print tensor.verify_grad(f, [x_val], rng=rng)



if __name__ == '__main__':
    test_unwrapper()
    exit(0)
    rng = np.random.RandomState(23455)

    x = tt.tensor4()
    emb = IdentityInput(x, size=2)
    conv = LeNetConvPoolLayer()
    conv.connect(emb, rng, ())


    X = tt.tensor3()
    y_time = tt.iscalar()
    y_seq_id = tt.iscalar()

    y = X[0:y_time, y_seq_id]

    f = theano.function([X, y_time, y_seq_id], y)



    exit(0)


    def step(xx, a):
        return xx

    x = tt.shared(np.random.randn(10, 1, 1))
    xf, _ = theano.scan(step, sequences=x, non_sequences=[1],
    go_backwards=False)
    xb, _ = theano.scan(step, sequences=x, non_sequences=[0], go_backwards=True)
    xb = xb[::-1,]
    diff = (xb - xf).norm(2)

    f = theano.function([], [x, xf, xb, diff])

    print f()

    exit(0)


    x = tt.shared(np.random.randn(10, 5))
    f = theano.function([], x[::-1,:])
    print f()
    print x.get_value()
    exit(0)


    test_verify_exprgrad()
    exit(0)
    import theano.gradient
    x = tt.scalar()
    f = theano.function([x], x)
    #print theano.gradient.numeric_grad(f, [0.0])
    theano.gradient.verify_grad(f, [[np.asarray(0.0)]], rng=np.random)
    exit(0)


    x = tt.shared(np.array([1, 2, 3]))
    b = tt.iscalar()
    y = tt.repeat(x.dimshuffle('x', 0), b, 0)

    f = theano.function([b], y)

    print f(5)
    exit(0)

    x = tt.shared(np.diag([1, 1, 1, 1, 1]))
    y = x[:, -1]
    y = tt.switch(tt.eq(y, 1), y * 10, 3)
    f = theano.function([], [x, y])
    print f()
    exit(0)
    #test_mlp()
    #test_cherrypick()
    v1 = tt.shared(np.random.randn(10, 5, 3))
    v2 = tt.shared(np.random.randn(10, 5, 1))

    a = tt.concatenate([v1, v2], axis=2)


    f = theano.function([], a)
    print f()
    print v1.get_value()
    print v2.get_value()
    exit(0)


    X = tt.tensor3()
    w = tt.matrix()
    b = tt.vector()
    f = theano.function([X, w, b], tt.dot(X, w) + b)

    out = f([
        [  # Timestep 1
               # dialog1           dialog 2
            [0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]
        ],
        [  # Timestep 2
            [0.2, 0.2, 0.2], [-0.2, -0.2, -0.2]
        ],
        [  # Timestep 3
            [0.3, 0.3, 0.3], [-0.3, -0.3, -0.3]
        ]
    ],
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        [1, 2, 3, 4, 5,]
    )

    import ipdb; ipdb.set_trace()


