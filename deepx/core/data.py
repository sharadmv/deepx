import numpy as np

from .node import ShapedNode
from .shape import Shape
from .. import util

class Data(ShapedNode):

    def __init__(self, shape, placeholder=None, name=None, is_input=True):
        super(Data, self).__init__([], [shape])
        self.placeholder = placeholder
        self.name = name
        self.is_input = is_input

    def get_placeholder(self):
        if self.placeholder is None:
            self.placeholder = self.shape.create_placeholder(name=self.name)
        return self.placeholder

    def inputs(self, *inputs):
        if self.is_input:
            return [self.get_placeholder()]
        return []

    @property
    def shape(self):
        return self.get_shapes_out()[0]

    @property
    def batch(self):
        return self.shape.batch

    def is_sequence(self):
        return self.shape.sequence

    @property
    def sequence(self):
        return self.shape.sequence

    def infer_shape(self):
        return

    def outputs(self):
        return [self.get_placeholder()]

    def __repr__(self):
        return "Data(%s)" % self.shape

    def copy(self, shape=None, placeholder=None, name=None, is_input=None):
        return Data(
            shape if shape is not None else self.shape,
            placeholder=placeholder if placeholder is not None else self.placeholder,
            name=name if name is not None else self.name,
            is_input=is_input if is_input is not None else self.is_input
        )

class Constant(Data):

    def __init__(self, value):
        value = np.array(value)
        super(Constant, self).__init__(Shape(value.shape), placeholder=value, is_input=False)

"""
class Data(Node):

    def __init__(self, shape,
                 placeholder=None,
                 datatype=None,
                 name=None,
                 is_input=True):
        if not isinstance(shape, Shape):
            raise TypeError("Must instantiate with Shape object.")
        super(Data, self).__init__()
        self.shape = shape
        self.placeholder = placeholder
        self.datatype = datatype or "Data"
        self.name = name
        self.is_input = is_input
        if placeholder is not None:
            self.placeholder = placeholder
        else:
            self.placeholder = self.shape.create_placeholder(name=self.name)

    def get_outputs(self, **kwargs):
        return [self]

    def get_graph_inputs(self):
        if self.is_input:
            return [self.placeholder]
        return []

    def get_graph_outputs(self):
        return [self.placeholder]

    def get_graph_parameters(self):
        return []

    def get_graph_updates(self, **kwargs):
        return []

    def reset_states(self):
        pass

    def reset_state(self, i):
        pass

    def initialize(self, **kwargs):
        if self.placeholder is None:
            self.placeholder = self.shape.create_placeholder(name=self.name)

    def reinitialize(self, **kwargs):
        self.initialize()
        self.shape.create_placeholder(name=self.name)

    # Shape inference

    def set_shapes_in(self, shapes_in):
        raise Exception("Should never be setting shapes_in of Data.")

    def set_shapes_out(self, shapes_out):
        raise Exception("Should never be setting shapes_out of Data.")

    def get_shapes_in(self):
        return []

    def get_shapes_out(self):
        return [self.shape]

    def get_num_inputs(self):
        return 0

    def get_num_outputs(self):
        return 1

    def infer_shape(self):
        pass

    def get_state(self, **kwargs):
        return None

    def set_state(self, state):
        pass

    def make_sequence(self, max_length):
        new_shape = self.shape.copy(sequence=True,
                                    max_length=max_length)
        return Data(new_shape, name=self.name, datatype=self.datatype)

    def get_placeholder(self):
        return self.placeholder

    def is_sequence(self):
        return self.shape.is_sequence()

    def get_batch_size(self):
        return self.shape.get_batch_size()

    @classmethod
    def concatenate(self, data_list):
        shapes = [d.get_shapes_out()[0] for d in data_list]
        tensor_list = [d.get_placeholder() for d in data_list]
        result = util.concatenate_data(tensor_list, sequence=any(d.is_sequence() for d in data_list))
        final_shape = Shape.concatenate(shapes)
        return Data(final_shape, placeholder=result)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        return Data(self.shape.copy(),
                    datatype=self.datatype,
                    name=self.name)

    def __str__(self):
        if self.name is not None:
            string = "{datatype}[{name}]({shape})".format(
                datatype=self.datatype,
                name=self.name,
                shape=self.shape
            )
        else:
            string = "{datatype}({shape})".format(
                datatype=self.datatype,
                shape=self.shape
            )
            if self.is_sequence():
                return "Sequence(%s)" % string
        return string

class Constant(Data):

    def __init__(self, value):
        value = np.array(value)
        super(Constant, self).__init__(Shape(value.shape), placeholder=value,
                                       datatype="Constant")

    def get_graph_inputs(self):
        return []
"""
