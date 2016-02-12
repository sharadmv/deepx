import numpy as np

from .. import backend as T

from .exceptions import ShapeException

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
        self.batch_size = None

        self.batch_size = None

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
            param = T.variable(np.zeros(shape)+value, name=name)
            self.parameters[name] = param
        else:
            param = T.variable(np.random.normal(size=shape) * 0.01, name=name)
            self.parameters[name] = param
        return param

    # Graph operations

    def chain(self, node):
        composite = CompositeNode(self, node)
        composite.infer_shape()
        return composite

    def get_parameter_value(self, name):
        return T.get_value(self.parameters[name])

    def set_parameter_value(self, name, value):
        T.set_value(self.parameters[name], value)

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

    def get_batch_size(self):
        return self.batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

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

    def freeze(self):
        node = self.copy()
        node.frozen = True
        return node

    def unfreeze(self):
        node = self.copy()
        node.frozen = False
        return node

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

    def recurrent_forward(self, X, **kwargs):
        return X.next(self._recurrent_forward(X.get_data(), **kwargs), self.get_shape_out())

    def _recurrent_forward(self, X, **kwargs):
        def step(input, _):
            return self._forward(input, **kwargs), []
        output = T.rnn(step, X, [])
        return output[1]

    def get_initial_states(self, X, shape_index=1):
        return None

    def reset_states(self):
        pass

    def step(self, X, state):
        return X.next(self._step(X.get_data(), state), self.get_shape_out()), None

    def _step(self, X, _):
        return self._forward(X)

    def forward(self, X, **kwargs):
        if X.is_sequence():
            return self.recurrent_forward(X)
        output = self._forward(X.get_data())
        return X.next(output, self.get_shape_out())

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

    def recurrent_forward(self, X, output, **kwargs):
        previous_left, previous_right = output
        left, left_out = self.left.recurrent_forward(X, previous_left, **kwargs)
        right, right_out = self.right.recurrent_forward(left, previous_right, **kwargs)
        return right, (left_out, right_out)

    def forward(self, X, **kwargs):
        return self.right.forward(self.left.forward(X, **kwargs), **kwargs)

    def _step(self, X, state):
        left_state, right_state = state
        left, left_state = self.left._step(X, left_state)
        right, right_state = self.right._step(left, right_state)
        return right, (left_state, right_state)

    def infer_shape(self):
        self.set_batch_size(self.left.get_batch_size())
        self.left.infer_shape()
        if self.left.get_shape_out() is not None:
            self.right.set_shape_in(self.left.get_shape_out())
            self.right.set_batch_size(self.left.get_batch_size())
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

    def get_batch_size(self):
        return self.left.get_batch_size()

    def set_batch_size(self, batch_size):
        self.left.set_batch_size(batch_size)
        self.right.set_batch_size(batch_size)

    def reset_states(self):
        self.left.reset_states()
        self.right.reset_states()

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

    def copy(self, keep_parameters=False):
        node = CompositeNode(self.left.copy(), self.right.copy())
        node.infer_shape()
        return node

    def get_initial_states(self, X, shape_index=1):
        return (self.left.get_initial_states(X, shape_index=shape_index),
                self.right.get_initial_states(X, shape_index=shape_index))

    def __str__(self):
        return "{left} >> {right}".format(
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
