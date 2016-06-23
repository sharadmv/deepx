from abc import ABCMeta, abstractmethod
import copy as cp

from .. import backend as T
from ..util import flatten
from .exceptions import ShapeOutError
from .shape import Shape

__all__ = ['Node', 'NodeList']

class Node(object):
    """
    The :class:`Node` is the highest level abstraction in DeepX.
    It represents anything that takes in a set of inputs
    and returns a set of outputs.
    """
    __metaclass__ = ABCMeta

    def __init__(self):

        self.shapes_in = None
        self.shapes_out = None

        self.frozen = False
        self.states = None
        self.updates = []

        self._predict = {}

    @abstractmethod
    def get_outputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_graph_inputs(self):
        pass

    def get_graph_outputs(self, *inputs, **kwargs):
        final_outputs = []
        dups = set()
        for output in self.get_outputs(*inputs, **kwargs):
            graph_outputs = output.get_graph_outputs()
            for graph_output in graph_outputs:
                if graph_output in dups:
                    continue
                final_outputs.append(graph_output)
                dups.add(graph_output)
        return final_outputs

    @abstractmethod
    def get_graph_parameters(self):
        pass

    @abstractmethod
    def get_graph_updates(self, **kwargs):
        pass

    @abstractmethod
    def reset_states(self):
        pass

    @abstractmethod
    def reset_state(self, i):
        pass

    @abstractmethod
    def initialize(self, **kwargs):
        pass

    @abstractmethod
    def reinitialize(self, **kwargs):
        pass

    # Shape inference

    @abstractmethod
    def set_shapes_in(self, shapes_in):
        pass

    @abstractmethod
    def set_shapes_out(self, shapes_out):
        pass

    @abstractmethod
    def get_shapes_in(self):
        pass

    @abstractmethod
    def get_shapes_out(self):
        pass

    @abstractmethod
    def get_num_inputs(self):
        pass

    @abstractmethod
    def get_num_outputs(self):
        pass

    @abstractmethod
    def infer_shape(self):
        pass

    @abstractmethod
    def get_state(self, **kwargs):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

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

    # Binary operations

    def chain(self, node):
        from .chain import Chain
        return Chain(self, node)

    def concat(self, node):
        from .ops import Concatenate
        return (self, node) >> Concatenate()

    def freeze(self):
        node = self.same()
        node.frozen = True
        return node

    def add(self, node):
        from .ops import Add
        from .data import Constant
        if not isinstance(node, Node):
            node = Constant(node)
        return (self, node) >> Add()

    def sub(self, node):
        from .ops import Sub
        from .data import Constant
        if not isinstance(node, Node):
            node = Constant(node)
        return (self, node) >> Sub()

    def rsub(self, node):
        from .ops import Sub
        from .data import Constant
        if not isinstance(node, Node):
            node = Constant(node)
        return (node, self) >> Sub()

    def prod(self, node):
        from .ops import Prod
        from .data import Constant
        if not isinstance(node, Node):
            node = Constant(node)
        return (self, node) >> Prod()

    def div(self, node):
        from .ops import Div
        from .data import Constant
        if not isinstance(node, Node):
            node = Constant(node)
        return (self, node) >> Div()

    def rdiv(self, node):
        from .ops import Div
        from .data import Constant
        if not isinstance(node, Node):
            node = Constant(node)
        return (node, self) >> Div()

    # Infix operations

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __rshift__(self, node):
        if isinstance(node, tuple):
            return self.chain(NodeList(node))
        return self.chain(node)

    def __rrshift__(self, node):
        if isinstance(node, tuple):
            return NodeList(node).chain(self)
        return self.chain(node)

    def __add__(self, node):
        return self.add(node)

    def __radd__(self, node):
        return self.add(node)

    def __mul__(self, node):
        return self.prod(node)

    def __rmul__(self, node):
        return self.prod(node)

    def __sub__(self, node):
        return self.sub(node)

    def __rsub__(self, node):
        return self.rsub(node)

    def __div__(self, node):
        return self.div(node)

    def __rdiv__(self, node):
        return self.rdiv(node)

    def __or__(self, node):
        return self.concat(node)

    def __getitem__(self, index):
        from .ops import Index
        return self.chain(Index(index))

    # Node Bookkeeping

    def get_updates(self):
        return self.updates

    def set_updates(self, updates):
        self.updates = updates

    def same(self):
        return cp.deepcopy(self)

    def copy(self):
        node = cp.deepcopy(self)
        node.reinitialize()
        return node

    def __repr__(self):
        return str(self)

    def __str__(self):
        return super(Node, self).__repr__()

class NodeList(Node):

    def __init__(self, nodes):
        super(NodeList, self).__init__()
        self.nodes = tuple(nodes)

    def get_outputs(self, *inputs, **kwargs):
        return tuple(flatten(node.get_outputs(*inputs, **kwargs) for node in self.nodes))

    def get_graph_inputs(self):
        inputs = []
        dups = set()
        for node in self.nodes:
            for input in node.get_graph_inputs():
                if input not in dups:
                    dups.add(input)
                    inputs.append(input)
        return inputs

    def get_graph_parameters(self):
        if self.frozen:
            return []
        # TODO: remove dups
        return [parameter for node in self.nodes for parameter in node.get_graph_parameters()]

    def get_graph_updates(self, **kwargs):
        # TODO: remove dups
        return [update for node in self.nodes for update in node.get_graph_updates()]

    def reset_states(self):
        for node in self.nodes:
            node.reset_states()

    def reset_state(self, i):
        for node in self.nodes:
            node.reset_state(i)

    def initialize(self, **kwargs):
        for node in self.nodes:
            node.initialize(**kwargs)

    def reinitialize(self, **kwargs):
        for node in self.nodes:
            node.initialize(**kwargs)

    # Shape inference

    def set_shapes_in(self, shapes_in):
        for node in self.nodes:
            node.set_shapes_in(shapes_in)

    def set_shapes_out(self, shapes_out):
        for node, s in zip(self.nodes, shapes_out):
            node.set_shapes_out(s)

    def get_shapes_in(self):
        return self.nodes[0].get_shapes_in()

    def get_shapes_out(self):
        return list(flatten(node.get_shapes_out() for node in self.nodes))

    def get_num_inputs(self):
        return self.nodes[0].get_num_inputs()

    def get_num_outputs(self):
        return sum(node.get_num_outputs() for node in self.nodes)

    def get_state(self, **kwargs):
        return tuple(node.get_state(**kwargs) for node in self.nodes)

    def set_state(self, state):
        for node, s in zip(self.nodes, state):
            node.set_state(s)

    def _infer(self, *args): pass

    def infer_shape(self):
        for node in self.nodes:
            node.infer_shape()

    def __repr__(self):
        return "NodeList([%s])" % ', '.join(map(repr, self.nodes))

    def __str__(self):
        return "(%s)" % ', '.join(map(str, self.nodes))
