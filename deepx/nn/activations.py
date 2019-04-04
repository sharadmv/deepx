import six
from abc import abstractmethod, ABCMeta

from deepx.core import Op
from deepx.nn.fc import FC
from deepx.backend import T

class Activation(Op):

    def __new__(cls, shape_in=None, shape_out=None, **kwargs):
        activation = super(Activation, cls).__new__(cls, **kwargs)
        if shape_in is None and shape_out is None:
            return activation
        activation.__init__(**kwargs)
        return FC(shape_in=shape_in, shape_out=shape_out) >> activation

    def shape_inference(self):
        self.set_shape_out(self.get_shape_in())

    def forward(self, *inputs, **kwargs):
        return [self.activate(X) for X in inputs]

    def is_initialized(self):
        return True

    @abstractmethod
    def activate(self, X):
        pass

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

class Softmax(Activation):

    def __new__(cls, *args, **kwargs):
        return super(Softmax, cls).__new__(cls, *args, **kwargs)

    def __init__(self, temperature=1.0):
        super(Softmax, self).__init__()
        self.temperature = temperature

    def activate(self, X):
        return T.softmax(X, self.temperature)

class Sigmoid(Activation):

    def activate(self, X):
        return T.sigmoid(X)

class Tanh(Activation):

    def activate(self, X):
        return T.tanh(X)

class Relu(Activation):

    def activate(self, X):
        return T.relu(X)

class Elu(Activation):

    def __init__(self, alpha=1.0):
        super(Elu, self).__init__()
        self.alpha = alpha

    def activate(self, X):
        return T.relu(X) + self.alpha * (T.exp((X - abs(X)) * 0.5) - 1)

class LeakyRelu(Activation):

    def __init__(self, alpha=1.0):
        super(LeakyRelu, self).__init__()
        self.alpha = alpha

    def activate(self, X):
        return T.relu(X, alpha=self.alpha)
