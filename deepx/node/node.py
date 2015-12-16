import theano
import theano.tensor as T
import numpy as np

from model import Model

class Node(object):

    def __init__(self):
        self.shape_in  = None
        self.shape_out = None

        self.parameters = {}
        self.frozen = False

    @property
    def shape(self):
        return (self.get_shape_in(), self.get_shape_out())

    # Node operations

    def infer_shape(self):
        shape_in = self.get_shape_in()
        if shape_in is not None:
            shape_out = self._infer(shape_in)
            self.set_shape_out(shape_out)
            self.initialize()

    def init_parameter(self, name, shape):
        param = theano.shared((np.random.normal(size=shape) * 0.01).astype(theano.config.floatX))
        self.parameters[name] = param
        return param

    # Graph operations

    def chain(self, node):
        composite = CompositeNode(self, node)
        composite.infer_shape()
        return composite

    def concatenate(self, node):
        concatenated = ConcatenatedNode(self, node)
        concatenated.infer_shape()
        return concatenated

    def create_model(self, mixins):
        if isinstance(mixins, tuple) or isinstance(mixins, list):
            return Model(self, mixins)
        return Model(self, [mixins])

    def get_parameters(self):
        if self.frozen:
            return []
        return self.parameters.values()

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

    def is_initialized(self):
        return not (None in self.shape)

    def set_shape_in(self, shape_in):
        if self.shape_in is None:
            self.shape_in = shape_in
        if self.shape_in != shape_in:
            raise Exception("Error inferring shape.")

    def set_shape_out(self, shape_out):
        if self.shape_out is None:
            self.shape_out = shape_out
        if self.shape_out != shape_out:
            raise Exception("Error inferring shape.")

    def get_shape_in(self):
        return self.shape_in

    def get_shape_out(self):
        return self.shape_out

    def set_state(self, state):
        assert self.is_initialized(), "Cannot set state of uninitialized node."
        for name, val in state.iteritems():
            self.parameters[name].set_value(val)

    def get_state(self):
        assert self.is_initialized(), "Cannot get state of uninitialized node."
        state = {}
        for name, val in self.parameters.iteritems():
            state[name] = val.get_value()
        return state

    def freeze_parameters(self):
        self.frozen = True

    def initialize(self):
        # No parameters for default node
        return

    def get_activation(self):
        raise Exception("Cannot get activation from single node.")

    def is_data(self):
        return False

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.get_shape_in(), self.get_shape_out())

    def __repr__(self):
        return str(self)

    # Abstract node methods

    def _infer(self, shape_in):
        raise NotImplementedError

    def forward(self, X):
        return Data(self._forward(X.get_data()), self.shape_out)

    def _forward(self, X):
        raise NotImplementedError

class CompositeNode(Node):

    def __init__(self, left, right):
        super(CompositeNode, self).__init__()
        self.left = left
        self.right = right

    def forward(self, X):
        return self.right.forward(self.left.forward(X))

    def infer_shape(self):
        self.left.infer_shape()
        if self.left.get_shape_out() is not None:
            self.right.set_shape_in(self.left.get_shape_out())
        self.right.infer_shape()

    def set_shape_in(self, shape_in):
        self.left.set_shape_in(shape_in)

    def set_shape_out(self, shape_out):
        self.right.set_shape_out(shape_out)

    def get_shape_in(self):
        return self.left.get_shape_in()

    def get_shape_out(self):
        return self.right.get_shape_out()

    def get_state(self):
        return (self.left.get_state(),
                self.right.get_state())

    def set_state(self, state):
        left_state, right_state = state
        self.left.set_state(left_state)
        self.right.set_state(right_state)

    def get_parameters(self):
        if self.frozen:
            return []
        return self.left.get_parameters() + self.right.get_parameters()

    def get_input(self):
        return self.left.get_input()

    def get_activation(self):
        return self.right.forward(self.left.get_activation())

    def __str__(self):
        return "{left} >> {right}".format(
            left=self.left,
            right=self.right
        )

class ConcatenatedNode(Node):

    def __init__(self, left, right):
        self.left, self.right = left, right

    def infer_shape(self):
        self.left.infer_shape()
        self.right.infer_shape()

    def get_shape_in(self):
        return [self.left.get_shape_in(), self.right.get_shape_out()]

    def get_shape_out(self):
        return self.left.get_shape_out() + self.right.get_shape_out()

    def forward(self, X):
        X1, X2 = X
        Y1, Y2 = self.left.forward(X1), self.right.forward(X2)
        return Y1.concat(Y2)

    def get_activation(self):
        Y1, Y2 = self.left.get_activation(), self.right.get_activation()
        return Y1.concat(Y2)

    def get_input(self):
        inputs = []
        input = self.left.get_input()
        if isinstance(input, list):
            inputs.extend(input)
        else:
            inputs.append(input)
        input = self.right.get_input()
        if isinstance(input, list):
            inputs.extend(input)
        else:
            inputs.append(input)
        return inputs

    def get_parameters(self):
        return self.left.get_parameters() + self.right.get_parameters()

    def __str__(self):
        return "[{left}; {right}]".format(
            left=self.left,
            right=self.right
        )

class Data(Node):

    def __init__(self, data, shape):
        super(Data, self).__init__()
        self.data = data
        self.shape_in = shape
        self.shape_out = shape

    def _infer(self, shape_in):
        return self.shape_out

    @property
    def ndim(self):
        return self.data.ndim

    def concat(self, data):
        my_data, other_data = self.get_data(), data.get_data()
        return Data(T.concatenate([my_data, other_data], axis=-1), self.shape_out + data.shape_out)

    def get_activation(self):
        return self

    def get_input(self):
        return self

    def get_data(self):
        return self.data

    def is_data(self):
        return True

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Data(%s, %s)" % (self.data, self.get_shape_out())
