from __future__ import print_function

import numpy as np
from sklearn.datasets import fetch_mldata

from deepx.nn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    mnist = fetch_mldata("MNIST original")

    X = mnist['data']
    N = X.shape[0]
    X = X.reshape((N, 1, 28, 28))
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

    conv_net = Image((1, 28, 28)) >> Conv((10, 2, 2), activation=Elu) >> Conv((20, 2, 2)) >> Flatten() >> Softmax(10)

    rmsprop = RMSProp(CrossEntropy(conv_net))

    def train(n_iter, lr):
        for i in range(n_iter):
            u = np.random.choice(np.arange(split))
            loss = rmsprop.train(Xtrain[u:u+50], ytrain[u:u+50], lr)
            print("Loss:", loss)

        # preds = conv_net.predict(Xtest).argmax(axis=1)
        # print("Error: ", 1 - (preds == labels[test_idx]).sum() / float(N - split))
