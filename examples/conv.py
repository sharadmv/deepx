from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata

from deepx.nn import *
from deepx import T

if __name__ == "__main__":
    mnist = fetch_mldata("MNIST original")

    X = mnist['data']
    N = X.shape[0]
    X = X.reshape((N, 28, 28, 1))
    labels = mnist['target']

    np.random.seed(0)
    idx = np.random.permutation(np.arange(70000))
    X = X[idx]
    labels = labels[idx].astype(np.int32)

    y = np.zeros((N, 10))
    for i in range(N):
        y[i, labels[i]] = 1

    split = int(0.9 * N)

    train_idx, test_idx = idx[:split], idx[split:]

    Xtrain, Xtest = X[train_idx], X[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]

    X_in = T.placeholder(T.floatx(), [None, 28, 28, 1])
    Y_in = T.placeholder(T.floatx(), [None, 10])

    conv_net = Conv((2, 2, 10)) >> Conv((2, 2, 20)) >> Flatten() >> Linear(10)
    logits = conv_net(X_in)
    predictions = T.argmax(logits, -1)
    loss = T.mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_in))

    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    sess = T.interactive_session()

    def train(n_iter, batch_size=20):
        for i in range(n_iter):
            idx = np.random.permutation(Xtrain.shape[0])[:batch_size]
            result = sess.run([loss, train_op], { X_in : Xtrain[idx], Y_in : ytrain[idx] })
            print("Loss:", result[0])

        preds = sess.run(predictions, { X_in : Xtest }).astype(np.int32)
        print("Error: ", 1 - (preds == labels[test_idx]).sum() / float(N - split))
    train(1000)
