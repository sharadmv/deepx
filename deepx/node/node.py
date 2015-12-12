import theano
import theano.tensor as T
import numpy as np

from exceptions import DimensionException
from model import Model

class Node(object):

    def __init__(self):
        self.shape_in  = None
        self.shape_out = None

        self.node_chain = [self]
        self.parameters = {}
        self._initialized = False
        self.input = None

    # Node operations

    def infer(self, shape_in=None):
        self.shape_in = self.shape_in or shape_in
        if self.shape_in is not None:
            self.shape_out = self._infer(self.shape_in)
        if None not in self.shape:
            self.init_parameters()
            self._initialized = True

    def init_parameter(self, name, shape):
        param = theano.shared(np.random.normal(size=shape) * 0.01)
        self.parameters[name] = param
        return param

    # Graph operations

    def chain(self, node):
        if node.is_data():
            raise Exception("Cannot chain to a data node.")
        new_node = Node()
        new_node.node_chain = self.node_chain + node.node_chain
        new_node.infer_shape()
        return new_node

    def create_model(self, mixins):
        if isinstance(mixins, tuple) or isinstance(mixins, list):
            return Model(self, mixins)
        return Model(self, [mixins])

    def infer_shape(self):
        shape_out = self.node_chain[0].shape_in
        for node in self.node_chain:
            node.infer(shape_out)
            shape_out = node.shape_out

    def get_parameters(self):
        params = []
        for node in self:
            params.extend(node.parameters.values())
        return params

    def copy(self):
        pass

    # Infix

    def __rshift__(self, node):
        return self.chain(node)

    def __or__(self, mixins):
        return self.create_model(mixins)

    # Getters and setters

    def get_input(self):
        input = self.node_chain[0]
        if not input.is_data():
            raise Exception("Cannot have chain without data at the beginning.")
        return [input]

    def get_shape_in(self):
        return self.shape_in

    def get_shape_out(self):
        return self.shape_out

    def get_in_nodes(self):
        return self.in_nodes

    def get_in_node(self):
        return list(self.in_nodes)[0]

    def get_out_node(self):
        return self.out_node

    def add_neighbor(self, node):
        self.neighbors.add(node)

    def get_neighbors(self):
        return self.neighbors

    def get_neighbor(self):
        if len(self.neighbors) == 0:
            return None
        return list(self.neighbors)[0]

    def __iter__(self):
        for node in self.node_chain:
            yield node

    def is_initialized(self):
        for node in self:
            if not node._initialized:
                return False
        return True

    def init_parameters(self):
        pass

    def get_activation(self):
        input = self.get_input()[0]
        X = input
        for node in self.node_chain[1:]:
            X = node.forward(X)
        return X

    def is_data(self):
        return False

    @property
    def shape(self):
        return (self.shape_in, self.shape_out)

    def __str__(self):
        return " >> ".join(n.to_str() for n in self)

    def __repr__(self):
        return str(self)

    def to_str(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.shape_in, self.shape_out)

    # Abstract node methods

    def _infer(self, shape_in):
        return None

    def forward(self, X):
        return Data(self._forward(X.get_data()), self.shape_out)

    def _forward(self, X):
        raise NotImplementedError

class Data(Node):

    def __init__(self, data, shape):
        super(Data, self).__init__()
        self.data = data
        self.shape_out = shape

    @property
    def ndim(self):
        return self.data.ndim

    def get_data(self):
        return self.data

    def is_data(self):
        return True

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return str(self)

    def to_str(self):
        return "%s(%s)" % (self.__class__.__name__,
                               self.shape_out)
