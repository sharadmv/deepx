import theano
import theano.tensor as T
import numpy as np

from model import Model

class Node(object):

    def __init__(self):
        self.shape_in  = None
        self.shape_out = None

        self.node_chain = [self]
        self.parameters = {}
        self.input = None

    @property
    def _initialized(self):
        return not (None in self.shape)

    # Node operations

    def infer(self, *inputs):
        shape_out = [a.shape_out for a in inputs]
        if None in shape_out:
            return self.shape_out
        return self._infer(*shape_out)

    def init_parameter(self, name, shape):
        param = theano.shared((np.random.normal(size=shape) * 0.01).astype(theano.config.floatX))
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

    def concatenate(self, node):
        new_node = ConcatenatedNode(self, node)
        new_node.infer_shape()
        return new_node

    def create_model(self, mixins):
        if isinstance(mixins, tuple) or isinstance(mixins, list):
            return Model(self, mixins)
        return Model(self, [mixins])

    def infer_shape(self):
        inputs = self.get_input()
        shape_out =  [a.shape_out for a in inputs]
        if len(shape_out) == 1:
            shape_out = shape_out[0]
        for node in self.node_chain:
            if node.shape_in is None:
                node.shape_in = shape_out
            shape_out = node.infer(*inputs)
            node.shape_out = shape_out
            inputs = [node]
            if node._initialized:
                node.init_parameters()

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

    def __add__(self, node):
        return self.concatenate(node)

    def __or__(self, mixins):
        return self.create_model(mixins)

    # Getters and setters

    def get_input(self):
        return self.node_chain[0].get_input()

    def get_shape_in(self):
        return self.node_chain[0].shape_out

    def get_shape_out(self):
        return self.node_chain[-1].shape_out

    def __iter__(self):
        for node in self.node_chain:
            yield node

    def is_initialized(self):
        for node in self:
            if not node._initialized:
                return False
        return True

    def init_parameters(self):
        # No parameters for default node
        return

    def get_activation(self):
        X = self.get_input()
        for node in self.node_chain:
            X = node.forward(*X)
            X = [X]
        return X[0]

    def is_data(self):
        return False

    @property
    def shape(self):
        return (self.shape_in, self.shape_out)

    def __getitem__(self, idx):
        return self.node_chain[idx]

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

    def forward(self, *args):
        return Data(self._forward(*[x.get_data() for x in args]), self.shape_out)

    def _forward(self, X):
        raise NotImplementedError


class ConcatenatedNode(Node):

    def __init__(self, left_chain, right_chain):
        super(ConcatenatedNode, self).__init__()
        self.left_chain, self.right_chain = left_chain, right_chain

    def _infer(self, shape_left, shape_right):
        return self.left_chain.get_shape_out() + self.right_chain.get_shape_out()

    def forward(self, X, Y):
        X = self.left_chain.get_activation()
        Y = self.right_chain.get_activation()
        return X.concat(Y)

    def get_input(self):
        return self.left_chain.get_input() + self.right_chain.get_input()

class Data(Node):

    def __init__(self, data, shape):
        super(Data, self).__init__()
        self.data = data
        self.shape_in = shape
        self.shape_out = shape

    def _infer(self, args):
        return args

    def forward(self, X):
        return X

    @property
    def ndim(self):
        return self.data.ndim

    def concat(self, data):
        my_data, other_data = self.get_data(), data.get_data()
        return Data(T.concatenate([my_data, other_data], axis=-1), self.shape_out + data.shape_out)

    def get_input(self):
        return [self]

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
