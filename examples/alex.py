import numpy as np
from sklearn.datasets import fetch_mldata
from deepx.nn import Image, Softmax, Tanh, Relu, Flatten, Conv, predict
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


    input = Image('X', (1, 28, 28))

    tower1 = input >> Conv((10, 2, 2)) >> Relu() >> Conv((20, 2, 2)) >> Flatten()
    tower2 = input >> Conv((10, 2, 2)) >> Relu() >> Conv((20, 2, 2)) >> Flatten()
    conv_net = (tower1 + tower2) >> Relu(256) >> Softmax(10)
    model = conv_net | (predict, rmsprop, cross_entropy)

    def train(n_iter, lr):
        for i in xrange(n_iter):
            u = np.random.choice(np.arange(split))
            loss = model.train(Xtrain[u:u+50], ytrain[u:u+50], lr)
            print "Loss[%u]: %f" % (i, loss)
        benchmark()

    def benchmark():
        preds = model.predict(Xtest).argmax(axis=1)
        print "Error: ", 1 - (preds == labels[test_idx]).sum() / float(N - split)
