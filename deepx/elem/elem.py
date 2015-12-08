from ..node import Node
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class Elem(Node)

    def __init__(self):
        super(Elem, self).__init__(None, None)

    def activate(self, X):
        return X

    def _forward(self, X):
        return self.activate(X)

class Tanh(Elem):

    def activate(self, X):
        return T.tanh(X)

class Sigmoid(Elem):

    def activate(self, X):
        return T.tanh(X)

class Relu(Elem):

    def activate(self, X):
        return T.nnet.relu(X)

class Dropout(Elem):

    def __init__(self, p):
        super(Dropout, self).__init__()
        self.p = p
        self.srng = RandomStreams()

    def activate(self, X):
        retain_prob = 1 - self.p
        X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
        return X

