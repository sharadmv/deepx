import numpy as np
from ..initialization import initialize_weights
from .. import T
import six
from abc import ABCMeta, abstractmethod

@six.add_metaclass(ABCMeta)
class Node(object):

    def __init__(self):
        self._initialized = False
        self.initialization = T.get_current_initialization()
        self.parameters = {}

    def __call__(self, *args):
        if self.get_dim_in() is None:
            self.infer_shape(*map(lambda x: T.get_shape(x)[1:], args))
        if self.is_initialized():
            self.initialize()
        else:
            raise Exception('Not enough information to initialize network')
        return self.forward(*args)

    def create_parameter(self, name, shape, initial_value=None):
        if name not in self.parameters:
            if initial_value is None:
                parameter = T.variable(
                    initialize_weights(self.initialization, shape),
                    name=name,
                )
            else:
                parameter = T.variable(
                    np.array(initial_value, dtype=T.floatx()),
                    name=name,
                )
            self.parameters[name] = parameter

    def get_parameter(self, name):
        return self.parameters[name]

    def get_parameters(self):
        assert self.is_initialized()
        keys = sorted(list(self.parameters.keys()))
        return [self.parameters[a] for a in keys]

    def get_parameter_list(self, *parameters):
        return [self.get_parameter(a) for a in parameters]

    def chain(self, right):
        from .chain import Chain
        return Chain(self, right)

    def add(self, right):
        from ..ops import Add
        return (self, right) >> Add()

    def __rshift__(self, right):
        return self.chain(right)

    def __rrshift__(self, right):
        from .chain import Chain
        if isinstance(right, tuple):
            from .node_list import NodeTuple
            return Chain(NodeTuple(right), self)
        return Chain(right, self)

    def __add__(self, right):
        return self.add(right)

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def infer_shape(self, *shapes_in):
        pass

    @abstractmethod
    def is_initialized(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_dim_in(self):
        pass

    @abstractmethod
    def get_dim_out(self):
        pass
