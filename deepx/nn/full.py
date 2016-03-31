from .. import backend as T

from ..node import ShapedNode

class Full(ShapedNode):

    def initialize(self):
        if not self.is_elementwise():
            self.init_parameter('W', (self.get_shape_in(), self.get_shape_out()))
            self.init_parameter('b', self.get_shape_out())

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
        W, b = self.parameters['W'], self.parameters['b']
        return self.activate(T.dot(X, W) + b)

    def __str__(self):
        if self.is_elementwise():
            return "%s()" % self.__class__.__name__
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.get_shape_in(), self.get_shape_out())
class Maxout(Full):


    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 4)
        super(Maxout, self).__init__(*args, **kwargs)

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.get_shape_in(), self.get_shape_out())
    def initialize(self):
        self.init_parameter('W', (self.k, self.get_shape_in(), self.get_shape_out()))
        self.init_parameter('b', (self.k, self.get_shape_out()))

    def activate(self, X):
        return T.max(X, axis=1)

class Softmax(Full):

    def __init__(self, *args, **kwargs):
        self.T = kwargs.pop('T', 1.0)
        super(Softmax, self).__init__(*args, **kwargs)

    def activate(self, X):
        return T.softmax(X, self.T)

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

class LeakyRelu(Full):

    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop('alpha', 0.1)
        super(LeakyRelu, self).__init__(*args, **kwargs)

    def activate(self, X):
        return T.relu(X, alpha=self.alpha)

class Tanlu(Full):

    def initialize(self):
        super(Tanlu, self).initialize()
        self.init_parameter('alpha', self.get_shape_out(), value=0.5)

    def activate(self, X):
        alpha = self.parameters['alpha']
        constrained_alpha = T.clip(alpha, 0, 1)
        return constrained_alpha * T.tanh(X) + (1 - constrained_alpha) * T.relu(X)
