import theano
import theano.tensor as T
import numpy as np

from model import Model
from ..util import create_tensor, pack_tuple, unpack_tuple

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

    def unroll(self, max_length=None):
        return SequenceNode(self, max_length=max_length)

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

    def __getitem__(self, idx):
        index_node = IndexNode(self, idx)
        index_node.infer_shape()
        return index_node

    def __rshift__(self, node):
        return self.chain(node)

    def __add__(self, node):
        return self.concatenate(node)

    def __or__(self, mixins):
        return self.create_model(mixins)

    def __call__(self, *args, **kwargs):
        return self.unroll(*args, **kwargs)

    # Getters and setters

    def is_recurrent(self):
        return False

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

    def recurrent_forward(self, X, output):
        if self.is_recurrent():
            out = self.forward(X, output)
            return out, out
        out = self.forward(X)
        return out, None

    def get_previous_zeros(self, N):
        return None

    def forward(self, X):
        return Data(self._forward(X.get_data()), self.get_shape_out())

    def _forward(self, X):
        raise NotImplementedError

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

    def get_previous_zeros(self, N):
        return (self.left.get_previous_zeros(N), self.right.get_previous_zeros(N))

    def __str__(self):
        return "{left} >> {right}".format(
            left=self.left,
            right=self.right
        )

class SequenceNode(Node):

    def __init__(self, node, max_length=None):
        super(SequenceNode, self).__init__()
        self.node = node
        self.max_length = max_length

        self.input = Sequence(self.node.get_input())

    def _infer(self, shape_in):
        return self.node.shape_out

    def forward(self, X):
        S = X.shape[0]
        N = X.shape[1]

        previous = self.get_previous_zeros(N)
        previous, shape = unpack_tuple(previous)

        def step(input, *previous):
            previous = pack_tuple(previous, shape)
            input = Data(input)
            output, previous = self.node.recurrent_forward(input, previous)
            return (output.get_data(),) + tuple(p.get_data() for p in unpack_tuple(previous)[0])

        output, updates = theano.scan(
            step,
            sequences=[X.get_data()],
            non_sequences=previous,
            n_steps=S,
        )
        return Data(output[0], self.get_shape_out())

    def get_input(self):
        return self.input

    def get_shape_in(self):
        return self.node.get_shape_in()

    def get_shape_out(self):
        return self.node.get_shape_out()

    def get_previous_zeros(self, N):
        return self.node.get_previous_zeros(N)

    def get_state(self):
        return self.node.get_state()

    def set_state(self, state):
        return self.node.set_state(state)

    def __str__(self):
        return "Sequence(%s)" % self.node

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

    def forward(self, X):
        return self.node.forward(X)[self.index]

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

    def __init__(self, data, shape=None):
        super(Data, self).__init__()
        self.data = data
        self.shape_in = shape
        self.shape_out = shape

    @property
    def shape(self):
        return self.data.shape

    def _infer(self, shape_in):
        return self.shape_out

    def forward(self, X):
        return X

    def __getitem__(self, idx):
        return self.index(idx)

    def index(self, idx):
        return Data(self.data[idx], self.shape_out)

    @property
    def ndim(self):
        return self.data.ndim

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
        return False

class Sequence(Data):

    def __init__(self, data_var):
        self.data_var = data_var
        self.sequence_dim = data_var.ndim

        self.data = create_tensor(self.sequence_dim + 1)
        self.shape_in = self.data_var.shape_in
        self.shape_out = self.data_var.shape_out

    def __str__(self):
        return "Sequence(%s)" % self.data_var

    def is_sequence(self):
        return True
