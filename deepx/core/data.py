from .layer import Layer
from .shape import Shape
from .. import util

class Data(Layer):

    def __init__(self, shape,
                 placeholder=None,
                 datatype=None,
                 name=None):
        if not isinstance(shape, Shape):
            raise TypeError("Must instantiate with Shape object.")
        super(Data, self).__init__()
        self.shape = shape
        self.placeholder = placeholder
        self.datatype = datatype or "Data"
        self.name = name
        self.initialize()

    def can_initialize(self):
        return True

    def make_sequence(self, max_length):
        new_shape = self.shape.copy(sequence=True,
                                    max_length=max_length)
        return Data(new_shape, name=self.name, datatype=self.datatype)

    def initialize(self):
        self.placeholder = self.placeholder or self.shape.create_placeholder(name=self.name)

    def _infer(self, shape_in):
        return shape_in

    def get_placeholder(self):
        return self.placeholder

    def get_shape_in(self):
        return [self.shape]

    def get_shape_out(self):
        return [self.shape]

    def is_sequence(self):
        return self.shape.is_sequence()

    def get_batch_size(self):
        return self.shape.get_batch_size()

    def get_inputs(self):
        return [self]

    def forward(self, *args, **kwargs):
        return [self]

    @classmethod
    def concatenate(self, data_list):
        shapes = [d.get_shape_out()[0] for d in data_list]
        tensor_list = [d.get_placeholder() for d in data_list]
        result = util.concatenate_data(tensor_list)
        final_shape = Shape.concatenate(shapes)
        return Data(final_shape, placeholder=result)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        return Data(self.get_shape_out()[0],
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
