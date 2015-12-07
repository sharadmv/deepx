import theano
import theano.tensor as T
import numpy as np
from theanify import Theanifiable

from exceptions import DimensionException, DataException
from data import Data
class Node(object):

    def __init__(self, n_in, n_out):
        self.n_in, self.n_out = n_in, n_out
        self.parameters = {}
        self.inputs = None
        self.output = None

    def add_input(self, input):
        if self.inputs is None:
            self.inputs = []
        self.inputs.append(input)

    def propagate(self):
        if self.get_input() is None:
            return
        self.output = self.forward(*self.inputs)

    def get_output(self):
        return self.output

    def get_input(self):
        return self.inputs

    def get_activations(self):
        return [l.get_output() for l in self]

    def _forward(self, *inputs):
        raise NotImplementedError

    def forward(self, *inputs):
        output = Data(self._forward(*(x.get_data() for x in inputs)))
        return output

    def chain(self, layer):
        return CompositeLayer(self, layer)

    def concat(self, layer):
        return ConcatenatedLayer(self, layer)

    # Operator API

    def __lshift__(self, layer):
        return layer.chain(self)

    def __rshift__(self, layer):
        return self.chain(layer)

    def __add__(self, layer):
        return self.concat(layer)

    def __or__(self, mixin):
        self.add_mixin(mixin.name, mixin)
        return self

    # Mixin API

    def add_mixin(self, name, mixin):
        self.mixins[name] = mixin
        mixin.setup(self)
        setattr(self, name, mixin.mix)
        return self

    def get_mixin(self, name):
        return self.mixins[name]

    def has_mixin(self, name):
        return name in self.mixins

    # General methods

    def alloc(self, N):
        return T.alloc(np.array(0).astype(theano.config.floatX), N, self.n_out)

    def __iter__(self):
        yield self

    def __getitem__(self, index):
        return list(self)[index]

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

class ConcatenatedLayer(Layer):

    def __init__(self, left_layer, right_layer):
        n_in = left_layer.n_in + right_layer.n_in
        n_out = left_layer.n_out + right_layer.n_out
        super(ConcatenatedLayer, self).__init__(n_in, n_out)

        self.left_layer = left_layer
        self.right_layer = right_layer

        self.propagate()

    def get_input(self):
        if self.left_layer.get_input() is None:
            return None
        if self.left_layer.get_input() is None:
            return None
        return self.left_layer.get_input() + self.right_layer.get_input()

    def propagate(self):
        if self.get_input() is None:
            return
        self.left_layer.propagate()
        self.right_layer.propagate()
        self.output = Data(T.concatenate([self.left_layer.output.get_data(),
                                          self.right_layer.output.get_data()]))

    def __str__(self):
        return "(%s) + (%s)" % (self.left_layer, self.right_layer)

class CompositeLayer(Layer):

    def __init__(self, in_layer, out_layer, *args, **kwargs):

        super(CompositeLayer, self).__init__(in_layer.n_in,
                                             out_layer.n_out, *args, **kwargs)

        self.in_layer = in_layer
        self.out_layer = out_layer

        if self.in_layer.n_out != self.out_layer.n_in:
            raise DimensionException(self.in_layer, self.out_layer)

        self.propagate()

    def __iter__(self):
        for layer in self.in_layer:
            yield layer

        for layer in self.out_layer:
            yield layer

    def get_output(self):
        return self.out_layer.get_output()

    def get_input(self):
        return self.in_layer.get_input()

    def add_input(self, input):
        self.in_layer.add_input(input)

    def propagate(self):
        if self.get_input() is None:
            return
        self.in_layer.propagate()
        self.out_layer.add_input(self.in_layer.get_output())
        self.out_layer.propagate()

    def get_parameters(self):
        return self.in_layer.get_parameters() + self.out_layer.get_parameters()

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


    def __len__(self):
        return len(self.in_layer) + len(self.out_layer)
