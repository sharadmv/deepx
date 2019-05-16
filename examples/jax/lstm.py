import tqdm
import numpy.random as npr
import time
import itertools
from deepx.backend import T
from deepx.nn import Softmax, Repeat
from deepx.rnn import LSTM

from jax import jit, grad
from jax.experimental import optimizers

num_epochs = 100
N = 1000
step_size = 1e-2
batch_size = 50
H = 50

net = LSTM(4, 50) >> Repeat(LSTM(200), 2) >> Softmax(2)
init_params = net.parameters

X = T.random_normal([N, H, 4])
Y = T.concat([T.zeros(N)[..., None], T.ones(N)[..., None]], -1)

@jit
def loss(params, data):
    X, Y = data
    Y_ = net(X, params=params)[:, -1]
    return T.mean(T.categorical_crossentropy(Y_, Y))

@jit
def accuracy(params, data):
    X, Y = data
    Y_ = net(X, params=params)[:, -1]
    target_class = T.argmax(Y, axis=-1)
    predicted_class = T.argmax(Y_, axis=-1)
    return T.mean(target_class == predicted_class, axis=-1)


num_train = X.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

def data_stream():
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield X[batch_idx], Y[batch_idx]
batches = data_stream()

opt_init, opt_update, get_params = optimizers.adam(step_size)

@jit
def update(i, opt_state, batch):
    params = get_params(opt_state)
    grad_loss = grad(loss)
    g = grad_loss(params, batch)
    return opt_update(i, g, opt_state)

opt_state = opt_init(init_params)
itercount = itertools.count()

print("\nStarting training...")
for epoch in range(num_epochs):
    params = get_params(opt_state)
    test_acc = accuracy(params, (X, Y))
    test_loss = loss(params, (X, Y))
    print("Test set accuracy {}".format(test_acc))
    print("Test set loss {}".format(test_loss))

    for _ in tqdm.trange(num_batches):
        opt_state = update(next(itercount), opt_state, next(batches))

params = get_params(opt_state)
test_acc = accuracy(params, (X, Y))
test_loss = loss(params, (X, Y))
print("Test set accuracy {}".format(test_acc))
print("Test set loss {}".format(test_loss))
