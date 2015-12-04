import numpy as np

from deepx.layer import *
from deepx.optimize import RMSProp

if __name__ == "__main__":
    # Creating dataset
    D, C = 1, 10
    np.random.seed(1)
    mu = np.random.normal(0, 5, size=C)
    X, y = [], []

    for i in xrange(1000):
        yi = np.random.choice(xrange(C))
        y.append(yi)
        X.append(np.random.normal(loc=mu[yi], scale=0.1))

    X, y = np.vstack(X), np.array(y)
    X, Xtest = X[:900], X[900:]
    labels, labels_test = y[:900], y[900:]

    y = np.zeros((900, C))

    for i in xrange(900):
        y[i, labels[i]] = 1

    H = 100

    mlp = Tanh(D, H) >> Softmax(H, C) <> CrossEntropy()

    rmsprop = RMSProp(mlp)
    mlp.compile_method('forward')

    def errors(X, labels):
        ypred = mlp.forward(X).argmax(axis=1)
        return (ypred != labels).sum()

    batch_size = 50
    iterations = 1000
    learning_rate = 100
    for i in xrange(iterations):
        u = np.random.randint(X.shape[0] - batch_size)
        print "Iteration %u: %f" % (i + 1, rmsprop.train(X[u:u+batch_size, :],
                                                            y[u:u+batch_size],
                                                            learning_rate))

    print "Training Error: %f" % (errors(X, labels) / float(900))
    print "Test Error: %f" % (errors(Xtest, labels_test) / float(100))
