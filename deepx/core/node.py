import numpy as np
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from functools import wraps

from .. import T
from ..util import flatten
from .exceptions import ShapeOutError
from .shape import Shape

# __all__ = ['Node', 'NodeList']

class DeviceDecorator(ABCMeta):
    def __init__(cls, name, bases, clsdict):
        if 'get_outputs' in clsdict:
            old = clsdict['get_outputs']
            @wraps(old)
            def new_get_outputs(self, *args, **kwargs):
                with T.device(self.device):
                    return old(self, *args, **kwargs)
            setattr(cls, 'get_outputs', new_get_outputs)

@add_metaclass(DeviceDecorator)
class Node(object):
    """
    The :class:`Node` is the highest level abstraction in DeepX.
    It represents anything that takes in a set of inputs
    and returns a set of outputs.
    """

    @abstractmethod
    def get_shapes_in(self):
        pass

    @abstractmethod
    def get_shapes_out(self):
        pass

    def get_num_inputs(self):
        return len(self.get_shapes_in())

    def get_num_outputs(self):
        return len(self.get_shapes_out())

    def get_outputs(self, inputs):
        if len(inputs) != self.get_num_inputs():
            raise Exception("shape mismatch")
        return self.forward(*inputs)

    def forward(self, *inputs):
        pass

    def __repr__(self):
        return "Node(%u, %u)" % (self.get_num_inputs(), self.get_num_outputs())

    def __rshift__(self, other):
        return self.chain(other)

    def __rrshift__(self, other):
        if not (isinstance(other, list) or isinstance(other, tuple)):
            raise Exception("not tuple or list")
        return [self.chain(c) for c in other]

    def chain(self, node):
        return Chain(self, node)

    @abstractmethod
    def infer_shape(self):
        pass


class ShapedNode(Node):

    def __init__(self, shapes_in, shapes_out):
        self.shapes_in = shapes_in
        self.shapes_out = shapes_out

    def get_shapes_in(self):
        return self.shapes_in

    def get_shapes_out(self):
        return self.shapes_out


class Data(ShapedNode):

    def __init__(self, shape, placeholder=None, name=None):
        super(Data, self).__init__([], [shape])
        self.placeholder = placeholder or shape.create_placeholder(name=name)

    @property
    def shape(self):
        return self.get_shapes_out()[0]

    def infer_shape(self):
        return

    def forward(self):
        return self.placeholder

class Constant(Data):

    def __init__(self, value):
        value = np.array(value)
        super(Constant, self).__init__(Shape(value.shape), placeholder=value)

    def get_graph_inputs(self):
        return []


class Chain(ShapedNode):

    def __init__(self, left, right):
        self.left = left
        self.right = right

        if self.left.get_num_outputs() != self.right.get_num_inputs():
            raise Exception("Shape mismatch")
        self.infer_shape()

    def get_shapes_in(self):
        return self.left.get_shapes_in()

    def get_shapes_out(self):
        return self.right.get_shapes_out()

    def set_shapes_in(self, shapes_in):
        self.left.set_shapes_in(shapes_in)

    def set_shapes_out(self, shapes_out):
        self.right.set_shapes_out(shapes_out)

    def infer_shape(self):
        self.left.infer_shape()
        self.right.infer_shape()
        left_shapes, right_shapes = [], []
        for left_shape, right_shape in zip(self.get_shapes_in(), self.get_shapes_out()):
            left_shape.unify(right_shape)

    def __repr__(self):
        return "%s >> %s" % (self.left, self.right)

def Scalar(**kwargs):
    return Data(Shape(()), **kwargs)
