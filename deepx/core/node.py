import copy as cp

from .. import backend as T
from .exceptions import ShapeOutError
from .shape import Shape

class Node(object):
    """
    The :class:`Node` is the highest level abstraction in DeepX.
    It represents anything that takes in a set of inputs
    and returns a set of outputs.
    """
    def __init__(self):

        self.shape_in = None
        self.shape_out = None
        self.frozen = False
        self.batch_size = None
        self.config = {}

        self.states = None
        self.updates = []

        self._predict = {}

    def get_updates(self):
        return self.updates

    def set_updates(self, updates):
        self.updates = updates

    def get_inputs(self):
        raise NotImplementedError

    def forward(self, inputs, **kwargs):
        raise NotImplementedError

    def get_outputs(self, **kwargs):
        self.initialize()
        return self.forward(self.get_inputs(), **kwargs)

    def get_graph_inputs(self):
        return [x.get_placeholder() for x in self.get_inputs()]

    def get_graph_outputs(self, **kwargs):
        return [x.get_placeholder() for x in self.get_outputs(**kwargs)]

    def get_graph_parameters(self):
        raise NotImplementedError

    def get_graph_updates(self, **kwargs):
        return self.get_updates()

    def predict(self, *args, **kwargs):
        dropout = kwargs.pop('dropout', False)
        if dropout not in self._predict:
            self.initialize()
            self._predict[dropout] = T.function(
                self.get_graph_inputs(),
                self.get_graph_outputs(dropout=dropout),
                updates=self.get_graph_updates()
            )
        return self._predict[dropout](*args, **kwargs)

    def get_initial_states(self, *args, **kwargs):
        return []

    def reset_states(self):
        if self.states is not None:
            for i in range(len(self.states)):
                self.reset_state(i)

    def reset_state(self, i):
        raise NotImplementedError

    def initialize(self, **kwargs):
        raise NotImplementedError

    def reinitialize(self, **kwargs):
        raise NotImplementedError

    # Shape inference

    def set_shape_in(self, shape_in):
        self.shape_in = shape_in

    def set_shape_out(self, shape_out):
        self.shape_out = shape_out

    def get_shape_in(self):
        return self.shape_in

    def get_shape_out(self):
        return self.shape_out

    def get_shape(self):
        return (self.get_shape_in(), self.get_shape_out())

    def infer_shape(self):
        shape_in = self.get_shape_in()
        shape_out = self.get_shape_out()
        if shape_in is not None:
            predicted_shape_out = self._infer(shape_in)
            if shape_out is None and shape_out == predicted_shape_out:
                self.set_shape_out(predicted_shape_out)
            elif shape_out != predicted_shape_out:
                raise ShapeOutError(self, shape_out)

    def _infer(self, shape_in):
        raise NotImplementedError

    # Binary operations

    def chain(self, node):
        from .ops import Chain
        return Chain(self, node)

    def concat(self, node):
        from .ops import Concatenate
        return Concatenate(self, node)

    def freeze(self):
        node = self.same()
        node.frozen = True
        return node

    # Infix operations

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __rshift__(self, node):
        return self.chain(node)

    def __or__(self, node):
        return self.concat(node)

    def __getitem__(self, index):
        from .ops import Index
        return self.chain(Index(index))

    # Node Bookkeeping

    def same(self):
        return cp.deepcopy(self)

    def copy(self):
        node = cp.deepcopy(self)
        if self.can_initialize():
            node.reinitialize()
        return node

    def __repr__(self):
        return str(self)

    def __str__(self):
        return super(Node, self).__repr__()
