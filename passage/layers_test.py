import numpy as np
import theano
import theano.tensor as tt

from layers import (Dense, IdentityInput, MatrixInput, MLP, CherryPick,
                             CrossEntropyObjective)
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


if __name__ == '__main__':
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


