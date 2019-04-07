import numpy as np
from deepx.backend import T
from deepx import nn, keras

network = (
    nn.Repeat(keras.Dense(10) >> keras.ReLU(), 4)
    >> nn.Repeat(nn.Relu(20), 4)
    >> nn.Softmax(10)
)

X = T.placeholder(T.floatx(), [None, 784])
Y = network(X)

sess = T.interactive_session()
result = sess.run(Y, {
    X: np.ones([1, 784])
})
