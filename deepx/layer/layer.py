import theano
import numpy as np

from exceptions import DimensionException
from model import Data, Model

class Layer(object):

    def __init__(self, n_in, n_out):
        self.n_in, self.n_out = n_in, n_out
        self.parameters = {}

    def __lshift__(self, layer):
        return CompositeLayer(layer, self)

    def __rshift__(self, layer):
        return CompositeLayer(self, layer)

    def __iter__(self):
        yield self

    def _forward(self, X):
        raise NotImplementedError

    def forward(self, X):
        return Data(self._forward(X.X))

    def _step(self, X):
        raise NotImplementedError

    def step(self, X):
        return Data(self._step(X.X))

    # General methods

    def initialize_weights(self, shape):
        return (np.random.standard_normal(size=shape) * 0.01).astype(theano.config.floatX)

    def init_parameter(self, name, shape):
        weights = self.initialize_weights(shape)
        self.parameters[name] = theano.shared(weights)
        return self.parameters[name]

    def is_composite(self):
        return False

    def get_parameter(self, name):
        return self.parameters[name]

    def set_parameter(self, name, value):
        return self.parameters[name].set_value(value)

    def get_parameters(self):
        return self.parameters.values()

    def __str__(self):
        return "%s(%u, %u)" % (self.__class__.__name__,
                               self.n_in, self.n_out)

    def __len__(self):
        return 1

class CompositeLayer(Layer):

    def __init__(self, in_layer, out_layer):

        self.in_layer = in_layer
        self.out_layer = out_layer

        if self.in_layer.n_out != self.out_layer.n_in:
            raise DimensionException(self.in_layer, self.out_layer)

        self.n_in = self.in_layer.n_in
        self.n_out = self.out_layer.n_out

    def is_composite(self):
        return True

    def __iter__(self):
        for layer in self.in_layer:
            yield layer

        for layer in self.out_layer:
            yield layer

    def get_parameters(self):
        return self.in_layer.get_parameters() + self.out_layer.get_parameters()

    def forward(self, X):
        return (X > self.in_layer) > self.out_layer

    def __str__(self):
        return "Composite({in_layer} >> {out_layer})".format(
            in_layer=str(self.in_layer),
            out_layer=str(self.out_layer)
        )

    def __len__(self):
        return len(self.in_layer) + len(self.out_layer)
