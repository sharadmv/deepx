import numpy as np

from deepx.layer import *
from deepx.optimize import rmsprop

if __name__ == "__main__":
    # Creating dataset
    D, C = 1, 2
    N = 10000
    np.random.seed(1)
    mu = np.random.normal(0, 10, size=C)
    X, y = [], []

    for i in xrange(N):
        yi = np.random.choice(xrange(C))
        y.append(yi)
        X.append(np.random.normal(loc=mu[yi], scale=0.1))

    X, y = np.vstack(X), np.array(y)
    split = int(0.9 * N)
    X, Xtest = X[:split], X[split:]
    labels, labels_test = y[:split], y[split:]

    y = np.zeros((split, C))

    for i in xrange(split):
        y[i, labels[i]] = 1

    H = 2

    mlp = Tanh(D, H) >> Softmax(H, C) | output | cross_entropy | rmsprop
    mlp.compile()

    def errors(X, labels):
        ypred = (mlp.output(X)).argmax(axis=1)
        return (ypred != labels).sum()

    batch_size = 50
    iterations = 1000
    learning_rate = 10
    for i in xrange(iterations):
        u = np.random.randint(X.shape[0] - batch_size)
        print "Iteration %u: %f" % (i + 1, mlp.train(X[u:u+batch_size, :],
                                                            y[u:u+batch_size],
                                                            learning_rate))

    print "Training Error: %f" % (errors(X, labels) / float(split))
    print "Test Error: %f" % (errors(Xtest, labels_test) / float(N - split))
