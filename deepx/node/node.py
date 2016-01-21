import numpy as np

from .. import backend as T

from .model import Model
from .exceptions import ShapeException
from ..util import pack_tuple, unpack_tuple

class Node(object):

    def __init__(self):
        self.shape_in  = None
        self.shape_out = None

        self.parameters = {}
        self.frozen = False

        self._initialized = False

        self._predict = None
        self._predict_dropout = None
        self.updates = []

    @property
    def shape(self):
        return (self.get_shape_in(), self.get_shape_out())

    # Passing forward through network

    def predict(self, *args, **kwargs):
        if self._predict is None:
            result = self.get_activation(use_dropout=False).get_data()
            self._predict = T.function(self.get_formatted_input(), [result], updates=self.get_updates())
        return self._predict(*args, **kwargs)

    def predict_with_dropout(self, *args, **kwargs):
        if self._predict_dropout is None:
            result = self.get_activation(use_dropout=True).get_data()
            self._predict_dropout = T.function(self.get_formatted_input(), [result])
        return self._predict_dropout(*args, **kwargs)

    def get_formatted_input(self):
        input = self.get_input()
        if not isinstance(input, list):
            input = [input]
        return [i.get_data() for i in input]

    def get_activation(self, use_dropout=True):
        return self.forward(self.get_input(), use_dropout=use_dropout)

    # Node operations

    def infer_shape(self):
        shape_in = self.get_shape_in()
        if shape_in is not None:
            shape_out = self._infer(shape_in)
            self.set_shape_out(shape_out)
            self.initialize()
            self._initialized = True

    def init_parameter(self, name, shape, value=None):
        if value:
            param = T.variable(np.zeros(shape)+value)
            self.parameters[name] = param
        else:
            param = T.variable(np.random.normal(size=shape) * 0.01)
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

    def get_parameters(self):
        if self.frozen:
            return []
        return list(self.parameters.values())

    def copy(self):
        raise NotImplementedError

    # Infix

    def __getitem__(self, idx):
        index_node = IndexNode(self, idx)
        index_node.infer_shape()
        return index_node

    def __rshift__(self, node):
        return self.chain(node)

    def __add__(self, node):
        return self.concatenate(node)

    def __call__(self, *args, **kwargs):
        return self.unroll(*args, **kwargs)

    # Getters and setters

    def get_updates(self):
        return self.updates

    def add_update(self, fro, to):
        self.updates.append((fro, to))

    def is_recurrent(self):
        return False

    def is_initialized(self):
        return not (None in self.shape)

    def set_shape_in(self, shape_in):
        if self.shape_in is None:
            self.shape_in = shape_in
        if self.shape_in != shape_in:
            raise ShapeException(self, shape_in)

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
        for name, val in state.items():
            T.set_value(self.parameters[name], val)

    def get_state(self):
        assert self.is_initialized(), "Cannot get state of uninitialized node."
        state = {}
        for name, val in self.parameters.items():
            state[name] = T.get_value(val).tolist()
        return state

    def freeze_parameters(self):
        self.frozen = True

    def initialize(self):
        # No parameters for default node
        return

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

    def recurrent_forward(self, X):
        def step(input, _):
            return self._forward(input), []

        _, output, _ = T.rnn(step, X.get_data(), [])
        return Data(output, self.get_shape_out(), sequence=True, batch_size=X.batch_size)

    def get_previous_zeros(self, N):
        return None

    def forward(self, X, **kwargs):
        if X.is_sequence():
            return self.recurrent_forward(X)
        return Data(self._forward(X.get_data()), self.get_shape_out(), batch_size=X.batch_size)

    def _forward(self, X):
        raise NotImplementedError

    def __eq__(self, node):
        if self.__class__ != node.__class__:
            return False
        if self.shape != node.shape:
            return False
        if self.get_state() != node.get_state():
            return False
        return True

class CompositeNode(Node):

    def __init__(self, left, right):
        super(CompositeNode, self).__init__()
        self.left = left
        self.right = right

    def recurrent_forward(self, X, output):
        previous_left, previous_right = output
        left, left_out = self.left.recurrent_forward(X, previous_left)
        right, right_out = self.right.recurrent_forward(left, previous_right)
        return right, (left_out, right_out)

    def forward(self, X, **kwargs):
        return self.right.forward(self.left.forward(X, **kwargs), **kwargs)

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

    def get_updates(self):
        return self.left.get_updates() + self.right.get_updates()

    def get_parameters(self):
        if self.frozen:
            return []
        return self.left.get_parameters() + self.right.get_parameters()

    def get_input(self):
        return self.left.get_input()

    def get_previous_zeros(self, N):
        return (self.left.get_previous_zeros(N), self.right.get_previous_zeros(N))

    def copy(self):
        node = CompositeNode(self.left.copy(), self.right.copy())
        node.infer_shape()
        return node

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
        return [self.left.get_shape_in(), self.right.get_shape_in()]

    def get_shape_out(self):
        if self.left.get_shape_out() is None:
            return None
        if self.right.get_shape_out() is None:
            return None
        return self.left.get_shape_out() + self.right.get_shape_out()

    def forward(self, X, collect_outputs=False):
        X1, X2 = X
        if collect_outputs:
            (left, left_out), (right, right_out) = self.left.forward(X1, True), self.right.forward(X2, True)
            return left.concat(right), (left_out, right_out)
        Y1, Y2 = self.left.forward(X1), self.right.forward(X2)
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

    def get_state(self):
        return (self.left.get_state(),
                self.right.get_state())

    def set_state(self, state):
        left_state, right_state = state
        self.left.set_state(left_state)
        self.right.set_state(right_state)

    def get_parameters(self):
        return self.left.get_parameters() + self.right.get_parameters()

    def __str__(self):
        return "[{left}; {right}]".format(
            left=self.left,
            right=self.right
        )

class IndexNode(Node):

    def __init__(self, node, index):
        super(IndexNode, self).__init__()
        self.node = node
        self.index = index

    def infer_shape(self):
        self.node.infer_shape()

    def get_shape_in(self):
        return self.node.get_shape_in()

    def get_shape_out(self):
        return self.node.get_shape_out()

    def forward(self, X, **kwargs):
        index = self.index
        out = self.node.forward(X, **kwargs)
        if index == -1:
            index = T.shape(out.get_data())[0] - 1
        return out[index, :, :]

    def get_input(self):
        return self.node.get_input()

    def get_state(self):
        return self.node.get_state()

    def set_state(self, state):
        self.node.set_state(state)

    def get_parameters(self):
        return self.node.get_parameters()

    def __str__(self):
        return "({node})[{index}]".format(
            node=self.node,
            index=self.index
        )

class Data(Node):

    def __init__(self, data, shape=None, sequence=False, batch_size=None):
        super(Data, self).__init__()
        self.data = data
        self.shape_in = shape
        self.shape_out = shape

        self.batch_size = batch_size

        self._is_sequence = sequence


    def _infer(self, shape_in):
        return self.shape_out

    def forward(self, X, **kwargs):
        return X

    def __getitem__(self, idx):
        return self.index(idx)

    def index(self, idx):
        return Data(self.data[idx], self.shape_out, batch_size=self.batch_size)

    @property
    def ndim(self):
        return T.ndim(self.data)

    def concat(self, data):
        my_data, other_data = self.get_data(), data.get_data()
        return Data(T.concatenate([my_data, other_data], axis=-1), self.shape_out + data.shape_out)

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

    def is_sequence(self):
        return self._is_sequence
