import numpy as np
from sklearn.datasets import fetch_mldata
from deepx.nn import Image, Softmax, Tanh, Flatten, Conv, predict
from deepx.optimize import rmsprop, cross_entropy

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
    for i in xrange(N):
        y[i, labels[i]] = 1

    split = int(0.9 * N)

    train_idx, test_idx = idx[:split], idx[split:]

    Xtrain, Xtest = X[train_idx], X[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]

    conv_net = Image('X', (1, 28, 28)) >> Conv((10, 2, 2)) >> Tanh() >> Conv((20, 2, 2)) >> Flatten() >> Tanh(128) >> Softmax(10) | (predict, cross_entropy, rmsprop)

    def train(n_iter, lr):
        for i in xrange(n_iter):
            u = np.random.choice(np.arange(split))
            loss = conv_net.train(Xtrain[u:u+50], ytrain[u:u+50], lr)
            print "Loss:", loss

        preds = conv_net.predict(Xtest).argmax(axis=1)
        print "Error: ", 1 - (preds == labels[test_idx]).sum() / float(N - split)
