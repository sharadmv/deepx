import theano
import numpy as np

from exceptions import DimensionException
from model import Data, Model

class FeedForwardLayer(object):

    def __init__(self, n_in, n_out):
        super(FeedForwardLayer, self).__init__()
        self.n_in, self.n_out = n_in, n_out
        self.parameters = {}

    def _forward(self, X):
        raise NotImplementedError

    def chain(self, X):
        if X.n_dim != self.n_in:
            raise DimensionException(X.n_dim, self.n_in)
        return Data(self.n_out, self._forward(X.X))

    def __lshift__(self, layer):
        return CompositeLayer(layer, self)

    def __rshift__(self, layer):
        return CompositeLayer(self, layer)

    def __ne__(self, loss):
        return Model(self, loss)

    def initialize_weights(self, shape):
        return (np.random.standard_normal(size=shape) * 0.01).astype(theano.config.floatX)

    def init_parameter(self, name, shape):
        weights = self.initialize_weights(shape)
        self.parameters[name] = theano.shared(weights)
        return self.parameters[name]

    def get_parameter(self, name):
        return self.parameters[name]

    def set_parameter(self, name, value):
        return self.parameters[name].set_value(value)

    def get_parameters(self):
        return self.parameters.values()

    def forward(self, X):
        X = Data(self.n_in, X)
        out = self.chain(X)
        return out.X

class CompositeLayer(FeedForwardLayer):

    def __init__(self, in_layer, out_layer):

        self.in_layer = in_layer
        self.out_layer = out_layer

        if self.in_layer.n_out != self.out_layer.n_in:
            raise DimensionException(self.in_layer.n_out, self.out_layer.n_in)

        self.n_in = self.in_layer.n_in
        self.n_out = self.out_layer.n_out

    def get_parameters(self):
        return self.in_layer.get_parameters() + self.out_layer.get_parameters()

    def chain(self, X):
        return (X > self.in_layer) > self.out_layer

    def __str__(self):
        return "{in_layer} >> {out_layer}".format(
            in_layer=str(self.in_layer),
            out_layer=str(self.out_layer)
        )
