import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from ..node import Node

class Elem(Node):

    def __init__(self):
        super(Elem, self).__init__(None, None)

    def activate(self, X):
        return X

    def is_elementwise(self):
        return True

    def _forward(self, X):
        return self.activate(X)

    def __str__(self):
        return "%s()" % self.__class__.__name__

class Tanh(Elem):

    def activate(self, X):
        return T.tanh(X)

class Sigmoid(Elem):

    def activate(self, X):
        return T.nnet.sigmoid(X)

class Relu(Elem):

    def activate(self, X):
        return T.nnet.relu(X)

class Dropout(Elem):

    def __init__(self, p):
        super(Dropout, self).__init__()
        self.p = p
        self.srng = RandomStreams()

    def get_activation(self, use_dropout=True):
        if use_dropout:
            return self.activation
        else:
            return self.inputs[0]

    def is_dropout(self):
        return True

    def activate(self, X):
        retain_prob = 1 - self.p
        X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
        return X

