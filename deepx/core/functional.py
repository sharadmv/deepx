from deepx import T
from abc import abstractmethod
from .node import Node

__all__ = ["FunctionalNode", "HOF"]

def HOF(func):
    def foo(node):
        return HigherOrderNode(node, func)
    return foo

class HigherOrderNode(Node):

    def __init__(self, node, func, **kwargs):
        super(HigherOrderNode, self).__init__(None, None, **kwargs)
        self.node = node
        self.func = func

    def infer_shape(self):
        self.node.set_shapes_in(self.get_shapes_in())
        self.node.infer_shape()
        self.set_shapes_out(self.node.get_shapes_out())

    def forward(self, X):
        return X

    def __repr__(self):
        return "HOF(%s)" % self.node

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

    def __rshift__(self, right):
        return self.chain(right)

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

# HOF = HigherOrderNode
