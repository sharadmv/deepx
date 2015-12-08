import numpy as np

from deepx.nn import Tanh, Softmax
from deepx.node import predict, cross_entropy, Matrix
from deepx.optimize import rmsprop

if __name__ == "__main__":
     # Creating dataset
    D, C = 1, 4
    np.random.seed(1)
    mu = np.random.normal(0, 10, size=C)
    X, y = [], []

    N = 10000

    for i in xrange(N):
        yi = np.random.choice(xrange(C))
        y.append(yi)
        X.append(np.random.normal(loc=mu[yi], scale=0.5))

    split = int(0.9 * N)
    X, labels = np.vstack(X), np.array(y)
    X, X_test = X[:split], X[split:]
    labels, labels_test = y[:split], y[split:]

    y = np.zeros((N, C))
    for i in xrange(split):
        y[i, labels[i]] = 1

    H = 400
    mlp = Matrix('X') >> Tanh(D, H) >> Tanh(H, H) >> Softmax(H, C) | (predict, cross_entropy, rmsprop)

    def errors(X, y):
        ypred = mlp.predict(X).argmax(axis=1)
        return 1 - (ypred == y).astype(np.int).mean()

    iterations = 100
    learning_rate = 10.0

    batch_size = 50
    for i in xrange(iterations):
        u = np.random.randint(X.shape[0] - batch_size)
        print "Iteration %u: %f" % (i + 1, mlp.train(X[u:u+batch_size, :],
                                                            y[u:u+batch_size],
                                                            learning_rate))

    # Evaluation

    print "Training error:", errors(X, labels)
    print "Test error:", errors(X_test, labels_test)
