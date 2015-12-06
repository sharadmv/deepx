import theano
import theano.tensor as T
import numpy as np
import logging
from theanify import Theanifiable

from exceptions import DimensionException, DataException
from model import Data

class Layer(Theanifiable):

    def __init__(self, n_in, n_out, mixins={}):
        self.n_in, self.n_out = n_in, n_out
        self.parameters = {}
        self.mixins = {}
        self.ndim = 2
        self.in_var = self.get_var(self.ndim)
        self._activation = None

        for name, mixin in mixins.iteritems():
            self.add_mixin(name, mixin)


    def __lshift__(self, layer):
        return CompositeLayer(layer, self)

    def __rshift__(self, layer):
        return CompositeLayer(self, layer)

    def add_mixin(self, name, mixin):
        logging.debug("Adding mixin: %s", name)
        self.mixins[name] = mixin
        mixin.setup(self)
        setattr(self, name, mixin.mix)
        return self

    def __or__(self, mixin):
        self.add_mixin(mixin.name, mixin)
        return self

    def __iter__(self):
        yield self

    def copy(self):
        return self.__class__(self.n_in, self.n_out, mixins=self.mixins.copy())

    def get_var(self, ndim):
        if ndim == 1:
            return T.matrix()
        if ndim == 2:
            return T.tensor3()
        if ndim == 3:
            return T.tensor4()
        raise Exception("Data too dimensional")

    def get_in_var(self):
        raise NotImplementedError

    def get_layer_var(self):
        raise NotImplementedError

    def _forward(self, X):
        raise NotImplementedError

    def forward(self, X, previous):
        return [Data(self._forward(X.get_data()))]

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

    def is_recurrent(self):
        return False

    def is_stateful(self):
        return False

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

        super(CompositeLayer, self).__init__(in_layer.n_in,
                                             out_layer.n_out)

        self.in_layer = in_layer
        self.out_layer = out_layer

        if self.in_layer.n_out != self.out_layer.n_in:
            raise DimensionException(self.in_layer, self.out_layer)

    def __iter__(self):
        for layer in self.in_layer:
            yield layer

        for layer in self.out_layer:
            yield layer

    def get_parameters(self):
        return self.in_layer.get_parameters() + self.out_layer.get_parameters()

    def forward(self, X, previous=None):
        activations = [X]
        if previous is not None and (type(previous) != list and type(previous) != tuple):
            raise DataException("Require previous activations to be iterable for multiple layers.")
        previous = previous or [None for _ in self]

        for layer, previous_activation in zip(self, previous):
            layer_below = activations[-1]
            activation = layer.forward(layer_below, previous_activation)
            activations.extend(activation)

        return activations[1:]

    def get_in_var(self):
        return list(self)[0].get_in_var()

    def get_layer_var(self):
        return [layer.get_layer_var() for layer in self]

    def is_composite(self):
        return True

    def is_recurrent(self):
        return [layer.is_recurrent() for layer in self]

    def is_stateful(self):
        return [layer.is_stateful() for layer in self]

    def __str__(self):
        return "Composite({layers})".format(
            layers=' >> '.join(map(str, self))
        )


    def copy(self):
        return self.__class__(self.in_layer.copy(), self.out_layer.copy())

    def __len__(self):
        return len(self.in_layer) + len(self.out_layer)
