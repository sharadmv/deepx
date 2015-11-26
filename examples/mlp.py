import logging
logging.basicConfig(level=logging.DEBUG)
import theano.tensor as T
from theanify import theanify, Theanifiable

from deepx.nn import MultilayerPerceptron
from deepx.optimize import RMSProp
from deepx.train import Trainer

import numpy as np

class MLPClassifier(Theanifiable):

    def __init__(self, D, H, C, n_layers):
        super(MLPClassifier, self).__init__()
        self.mlp = MultilayerPerceptron('mlp', D, H, C, n_layers=n_layers)
        self.compile_method('errors')

    @theanify(T.matrix('X'), T.ivector('y'))
    def cost(self, X, y):
        ypred = self.mlp.forward(X)
        return T.nnet.categorical_crossentropy(ypred, y).mean()

    @theanify(T.matrix('X'), T.ivector('y'))
    def errors(self, X, y):
        y_pred = self.mlp.forward(X).argmax(axis=1)
        return T.mean(T.neq(y_pred, y))

    def get_parameters(self):
        return self.mlp.get_parameters()

if __name__ == "__main__":
    # Creating dataset
    D, C = 1, 4
    np.random.seed(1)
    mu = np.random.normal(0, 5, size=C)
    X, y = [], []

    for i in xrange(1000):
        yi = np.random.choice(xrange(C))
        y.append(yi)
        X.append(np.random.normal(loc=mu[yi], scale=0.5))

    X, y = np.vstack(X), np.array(y)
    X, Xtest = X[:900], X[900:]
    y, ytest = y[:900], y[900:]


    H = 100
    L = 1
    mlp = MLPClassifier(D, H, C, L)
    rmsprop = RMSProp(mlp)

    # Training

    iterations = 100
    learning_rate = 100.0

    batch_size = 50
    for i in xrange(iterations):
        u = np.random.randint(X.shape[0] - batch_size)
        print "Iteration %u: %f" % (i + 1, rmsprop.train(X[u:u+batch_size, :],
                                                            y[u:u+batch_size],
                                                            learning_rate))

    # Evaluation

    print "Training error:", mlp.errors(X, y)
    print "Test error:", mlp.errors(Xtest, ytest)
