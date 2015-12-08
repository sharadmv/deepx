import theano
import theano.tensor as T
import numpy as np

from data import Data
from model import Model
from exceptions import DimensionException

class Node(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

        self.inputs = []
        self.activation = None

        self.parameters = {}

    def forward(self, *inputs):
        return Data(self._forward(*(i.get_data() for i in self.inputs)))

    def _forward(self, *inputs):
        raise NotImplementedError

    def propagate(self):
        if self.has_input():
            self.set_activation(self.forward(self.inputs))

    def chain(self, node):
        return CompositeNode(self, node)

    def concat(self, node):
        return ConcatenateNode(self, node)

    def add_input(self, data):
        self.inputs.append(data)

    def set_activation(self, activation):
        self.activation = activation

    def has_input(self):
        return len(self.inputs) > 0

    def has_activation(self):
        return self.get_activation() is not None

    def get_inputs(self):
        return self.inputs

    def get_activation(self):
        return self.activation

    def create_model(self, mixins):
        if isinstance(mixins, tuple):
            return Model(self, mixins)
        elif isinstance(mixins, list):
            return Model(self, mixins)
        else:
            return Model(self, (mixins,))

    # Parameter methods

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

    # Infix methods

    def __rshift__(self, node):
        return self.chain(node)

    def __or__(self, mixins):
        return self, self.create_model(mixins)

    def __add__(self, node):
        return self.concat(node)

    # General methods

    def is_data(self):
        return False

    def __iter__(self):
        yield self

    def __str__(self):
        return "%s(%u, %u)" % (self.__class__.__name__,
                               self.n_in, self.n_out)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return 1

class CompositeNode(Node):

    def __init__(self, in_node, out_node):
        super(CompositeNode, self).__init__(in_node.n_in, out_node.n_out)
        self.in_node = in_node
        self.out_node = out_node

        if self.out_node.n_in != self.in_node.n_out:
            raise DimensionException(self.in_node, self.out_node)

        self.propagate()

    def add_input(self, data):
        self.in_node.add_input(data)

    def propagate(self):
        self.in_node.propagate()
        if self.in_node.has_activation():
            self.out_node.inputs = [self.in_node.get_activation()]
        self.out_node.propagate()

    def get_inputs(self):
        return self.in_node.get_inputs()

    def get_activation(self):
        return self.out_node.get_activation()

    def get_parameters(self):
        return self.in_node.get_parameters() + self.out_node.get_parameters()

    def __iter__(self):
        for node in self.in_node:
            yield node
        for node in self.out_node:
            yield node

    def __str__(self):
        return "%s >> %s" % (self.in_node,
                             self.out_node)

class ConcatenateNode(Node):

    def __init__(self, left_node, right_node):
        n_in = left_node.n_out + right_node.n_out
        super(ConcatenateNode, self).__init__(n_in,
                                              n_in)
        self.left_node = left_node
        self.right_node = right_node

        self.left_node.propagate()
        self.right_node.propagate()
        self.inputs = [self.left_node.get_activation(), self.right_node.get_activation()]

        self.propagate()

    def _forward(self, *inputs):
        return T.concatenate(inputs, axis=1)

    def get_inputs(self):
        return self.left_node.get_inputs() + self.right_node.get_inputs()

    def get_parameters(self):
        return self.left_node.get_parameters() + self.right_node.get_parameters()

    def __str__(self):
        return "(%s) + (%s)" % (self.left_node, self.right_node)
