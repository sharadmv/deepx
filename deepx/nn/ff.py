import theano.tensor as T

from ..node import Node

class Linear(Node):
    def __init__(self, n_in=None, n_out=None):
        super(Linear, self).__init__()

        self._elementwise = False
        if n_in is None:
            self._elementwise = True
        else:
            if n_out is not None:
                self.shape_in = n_in
                self.shape_out = n_out
            else:
                self.shape_out = n_in

    def initialize(self):
        if not self.is_elementwise():
            self.W = self.init_parameter('W', (self.shape_in, self.shape_out))
            self.b = self.init_parameter('b', self.shape_out)

    def is_elementwise(self):
        return self._elementwise

    def _infer(self, shape_in):
        if self.is_elementwise():
            return shape_in
        return self.shape_out

    def activate(self, X):
        return X

    def _forward(self, X):
        if self.is_elementwise():
            return self.activate(X)
        return self.activate(T.dot(X, self.W) + self.b)

    def to_str(self):
        if self.is_elementwise():
            return "%s()" % self.__class__.__name__
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.shape_in, self.shape_out)

class Softmax(Linear):

    def activate(self, X):
        if X.ndim <= 2:
            e_x = T.exp((X - X.max(axis=1)[:, None]))
            return e_x / e_x.sum(axis=1)[:, None]
        e_x = T.exp((X - X.max(axis=2)[:, :, None]))
        return e_x / e_x.sum(axis=2)[:, :, None]

class Sigmoid(Linear):

    def activate(self, X):
        return T.nnet.sigmoid(X)

class Tanh(Linear):

    def activate(self, X):
        return T.tanh(X)

class Relu(Linear):

    def activate(self, X):
        return X * (X > 0)
