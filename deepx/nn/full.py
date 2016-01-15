from .. import backend as T

from ..node import Node

class Full(Node):
    def __init__(self, n_in=None, n_out=None):
        super(Full, self).__init__()

        self._elementwise = False
        if n_in is None and n_out is None:
            self._elementwise = True
        else:
            if n_out is not None:
                self.shape_in = n_in
                self.shape_out = n_out
            else:
                self.shape_out = n_in

        self.infer_shape()

    def initialize(self):
        if self._initialized:
            return
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
        if self.is_elementwise():
            raise Exception("No identity nodes allowed.")
        return X

    def _forward(self, X):
        if self.is_elementwise():
            return self.activate(X)
        return self.activate(T.dot(X, self.W) + self.b)

    def copy(self):
        return self.__class__(self.get_shape_in(), self.get_shape_out())

    def __str__(self):
        if self.is_elementwise():
            return "%s()" % self.__class__.__name__
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.get_shape_in(), self.get_shape_out())

class Softmax(Full):

    def activate(self, X):
        return T.softmax(X)

class Sigmoid(Full):

    def activate(self, X):
        return T.sigmoid(X)

class Tanh(Full):

    def activate(self, X):
        return T.tanh(X)

class Relu(Full):

    def activate(self, X):
        return T.relu(X)

class Elu(Full):

    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop('alpha', 1.0)
        super(Elu, self).__init__(*args, **kwargs)

    def activate(self, X):
        return T.relu(X) + self.alpha * (T.exp((X - abs(X)) * 0.5) - 1)

    def copy(self):
        return self.__class__(self.get_shape_in(), self.get_shape_out(), alpha=self.alpha)

class LeakyRelu(Full):

    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop('alpha', 0.1)
        super(LeakyRelu, self).__init__(*args, **kwargs)

    def activate(self, X):
        return T.relu(X, alpha=self.alpha)

    def copy(self):
        return self.__class__(self.get_shape_in(), self.get_shape_out(), alpha=self.alpha)

class Tanlu(Full):

    def __init__(self, *args, **kwargs):
        super(Tanlu, self).__init__(*args, **kwargs)
        self.alpha = self.init_parameter('alpha', self.get_shape_out(), value=0.5)

    def activate(self, X):
        constrained_alpha = T.clip(self.alpha, 0, 1)
        return constrained_alpha * T.tanh(X) + (1 - constrained_alpha) * T.relu(X)
