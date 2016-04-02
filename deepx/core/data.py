from .. import backend as T
from .node import ShapedNode

class Data(ShapedNode):

    def __init__(self, dim,
                 datatype=None,
                 name=None,
                 sequence=False,
                 max_length=None,
                 batch_size=None):

        assert isinstance(dim, int) or isinstance(dim, tuple)

        if isinstance(dim, int):
            dim = (dim,)

        self.dim = dim
        self.name = name
        self.datatype = datatype or "Data"
        self.sequence = sequence
        self.batch_size = batch_size
        self.max_length = max_length

        if self.sequence:
            self.placeholder = T.placeholder(shape=[self.max_length, self.batch_size] + list(self.dim), name=self.name)
        else:
            self.placeholder = T.placeholder(shape=[self.batch_size] + list(self.dim), name=self.name)
        super(Data, self).__init__(dim, dim)

    def get_inputs(self):
        return [self]

    def forward(self, *args, **kwargs):
        return [self]

    def has_parameters(self):
        return False

    @classmethod
    def from_placeholder(cls, placeholder, dim, batch_size, **kwargs):
        data = Data(dim,
                    datatype=kwargs.get('datatype', None),
                    sequence=kwargs.get('sequence', False),
                    max_length=kwargs.get('max_length', None),
                    batch_size=batch_size)
        data.placeholder = placeholder
        return data

    def make_sequence(self, max_length):
        return Data(self.dim,
                    datatype=self.datatype,
                    name=self.name,
                    sequence=True,
                    max_length=max_length,
                    batch_size=self.batch_size)

    def get_parameter_tree(self):
        return None

    def set_parameter_tree(self, params):
        pass

    def get_placeholder(self):
        return self.placeholder

    def get_shape_in(self):
        if len(self.dim) == 1:
            return self.dim[0]
        return self.dim

    def get_shape_out(self):
        if len(self.dim) == 1:
            return self.dim[0]
        return self.dim

    def is_sequence(self):
        return self.sequence

    def is_input(self):
        return True

    def copy(self, **kwargs):
        return Data(self.dim,
                    datatype=self.datatype,
                    name=self.name,
                    sequence=self.sequence,
                    max_length=self.max_length,
                    batch_size=self.batch_size)

    def get_state(self, **kwargs):
        return None

    def set_state(self, state):
        pass

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.name is not None:
            if self.sequence:
                return "Sequence[%s](%s(%s))" % (self.name, self.datatype, str(self.get_shape_out()))
            return "%s[%s](%s)" % (self.datatype, self.name, str(self.get_shape_out()))
        else:
            if self.sequence:
                return "Sequence(%s(%s))" % (self.datatype, str(self.get_shape_out()))
            return "%s(%s)" % (self.datatype, str(self.get_shape_out()))
