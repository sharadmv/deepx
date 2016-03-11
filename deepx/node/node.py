import numpy as np

from .. import backend as T

from .exceptions import ShapeException

class Node(object):

    def __init__(self, *args, **kwargs):

        self.args = (args, kwargs)

        self.shape_in  = None
        self.shape_out = None
        self.batch_size = None

        self.parameters = {}
        self.frozen = False

        self.initialized = False

        self._predict = None
        self._predict_dropout = None

        self.updates = []

    @property
    def shape(self):
        return (self.get_shape_in(), self.get_shape_out())

    def infer_shape(self):
        shape_in = self.get_shape_in()
        shape_out = None
        if shape_in is not None:
            shape_out = self._infer(shape_in)
        self.set_shape_out(shape_out)
        if not self.is_initialized() and self.can_initialize():
            self.initialize()
            self.initialized = True

    def can_initialize(self):
        raise NotImplementedError

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

    def init_parameter(self, name, shape, value=None):
        if value:
            param = T.variable(np.zeros(shape)+value, name=name)
            self.parameters[name] = param
        else:
            param = T.variable(np.random.normal(size=shape) * 0.1, name=name)
            self.parameters[name] = param
        return param

    # Graph operations

    def chain(self, node):
        return CompositeNode(self, node)

    def get_parameter_value(self, name):
        return T.get_value(self.parameters[name])

    def set_parameter_value(self, name, value):
        T.set_value(self.parameters[name], value)

    def get_parameters(self):
        if self.frozen:
            return []
        return list(self.parameters.values())

    def copy(self, keep_parameters=False):
        args, kwargs = self.args
        node = self.__class__(*args, **kwargs)
        node.set_shape_in(self.shape_in)
        node.set_shape_out(self.shape_out)
        node.infer_shape()
        node.set_initialized(self.is_initialized())
        if keep_parameters:
            node.parameters = self.parameters
        return node

    # Infix

    def __getitem__(self, idx):
        index_node = IndexNode(self, idx)
        return index_node

    def __rshift__(self, node):
        return self.chain(node)

    # Getters and setters

    def get_updates(self):
        return self.updates

    def add_update(self, fro, to):
        self.updates.append((fro, to))

    def is_recurrent(self):
        return False

    def is_initialized(self):
        return self.initialized

    def set_initialized(self, initialized):
        self.initialized = initialized

    def set_shape_in(self, shape_in):
        if self.shape_in is None:
            self.shape_in = shape_in
        if self.shape_in != shape_in:
            raise ShapeException(self, shape_in)

    def set_shape_out(self, shape_out):
        if shape_out is None:
            return
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
        assert self.is_initialized(), "Cannot get state of uninitialized node %s." % self
        state = {}
        for name, val in self.parameters.items():
            state[name] = T.get_value(val).tolist()
        return state

    def freeze(self):
        node = self.copy(keep_parameters=True)
        node.frozen = True
        return node

    def unfreeze(self):
        node = self.copy(keep_parameters=True)
        node.frozen = False
        return node

    def tie(self, node):
        new_node = self.copy(keep_parameters=True)
        for key, val in node.parameters.items():
            new_node.parameters[key] = val
        return new_node

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

    def reset_state(self, i):
        pass

    def step(self, X, state):
        out, state = self._step(X.get_data(), state)
        return X.next(out, self.get_shape_out()), state

    def _step(self, X, _):
        return self._forward(X), None

    def forward(self, X, **kwargs):
        if X.is_sequence():
            return self.recurrent_forward(X)
        output = self._forward(X.get_data())
        return X.next(output, self.get_shape_out())

    def _forward(self, X):
        raise NotImplementedError("_forward not implemented for %s" % str(self))

    def __eq__(self, node):
        if self.__class__ != node.__class__:
            return False
        if self.shape != node.shape:
            return False
        if self.get_state() != node.get_state():
            return False
        return True

class ShapedNode(Node):

    def __init__(self, *args, **kwargs):
        super(ShapedNode, self).__init__(*args, **kwargs)
        assert len(args) <= 2
        self._elementwise = False
        if len(args) == 2:
            shape_in, shape_out = args
        elif len(args) == 1:
            shape_in, shape_out = None, args[0]
        else:
            shape_in, shape_out = None, None
            self._elementwise = True
        self.set_shape_in(shape_in)
        self.set_shape_out(shape_out)
        if self.can_initialize():
            self.initialize()
            self.initialized = True

    def can_initialize(self):
        return (self.get_shape_in() is not None) and (self.get_shape_out() is not None)

class CompositeNode(Node):

    def __init__(self, left, right):
        super(CompositeNode, self).__init__()
        self.left = left
        self.right = right
        self.infer_shape()

    def is_initialized(self):
        return self.left.is_initialized() and self.right.is_initialized()

    def recurrent_forward(self, X, **kwargs):
        left = self.left.recurrent_forward(X, **kwargs)
        right = self.right.recurrent_forward(left, **kwargs)
        return right

    def forward(self, X, **kwargs):
        return self.right.forward(self.left.forward(X, **kwargs), **kwargs)

    def step(self, X, state):
        left_state, right_state = state
        left, left_state = self.left.step(X, left_state)
        right, right_state = self.right.step(left, right_state)
        return right, (left_state, right_state)

    def infer_shape(self):
        self.left.infer_shape()
        self.set_batch_size(self.get_batch_size())
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

    def set_batch_size(self, batch_size):
        self.left.set_batch_size(batch_size)
        self.right.set_batch_size(batch_size)

    def get_batch_size(self):
        return self.left.get_batch_size()

    def get_state(self):
        return (self.left.get_state(),
                self.right.get_state())

    def reset_states(self):
        self.left.reset_states()
        self.right.reset_states()

    def reset_state(self, i):
        self.left.reset_state(i)
        self.right.reset_state(i)

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

    def copy(self, **kwargs):
        node = CompositeNode(self.left.copy(**kwargs), self.right.copy(**kwargs))
        node.infer_shape()
        return node

    def tie(self, node):
        new_node = CompositeNode(self.left.tie(node.left), self.right.tie(node.right))
        new_node.infer_shape()
        return new_node


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
