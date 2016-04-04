from .. import backend as T
from .node import ShapedNode

class Data(ShapedNode):

    def __init__(self, dim,
                 datatype=None,
                 name=None,
                 sequence=False,
                 max_length=None,
                 batch_size=None,
                 dtype=T.floatx()):

        assert isinstance(dim, int) or isinstance(dim, tuple)

        if isinstance(dim, int):
            dim = (dim,)


        self.dim = dim
        super(Data, self).__init__(dim, dim)

        self.name = name
        self.datatype = datatype or "Data"
        self.sequence = sequence
        self.batch_size = batch_size
        self.max_length = max_length
        self.dtype = dtype

        if self.sequence:
            self.placeholder = T.placeholder(shape=[self.max_length, self.batch_size] + list(self.dim), name=self.name, dtype=dtype)
        else:
            self.placeholder = T.placeholder(shape=[self.batch_size] + list(self.dim), name=self.name, dtype=dtype)

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

    @classmethod
    def convert_shape_to_tuple(cls, dim):
        if isinstance(dim, tuple):
            return dim
        return (dim,)

    @classmethod
    def tile_data(cls, from_data, to_data):
        dim = 0
        while T.ndim(from_data) < T.ndim(to_data):
            from_data = T.expand_dims(from_data, 0)
            to_dim = T.shape(to_data)[dim]
            from_data = T.tile(from_data, [to_dim] + [1] * (T.ndim(from_data) - 1))
            dim += 1
        return from_data

    @classmethod
    def concatenate(cls, datas):
        shapes = [data.get_shape_out() for data in datas]
        max_data = sorted([d.get_placeholder() for d in datas], key=lambda x: T.ndim(x))[::-1]

        raw_datas = max_data[0:1] + [Data.tile_data(d, max_data[0]) for d in max_data[1:]]

        return Data.from_placeholder(
            T.concatenate(raw_datas, axis=-1),
            sum(shapes),
            datas[0].batch_size,
            sequence=any(d.is_sequence() for d in datas)
        )

    @classmethod
    def index(cls, data, index):
        raw_data = data.get_placeholder()
        return Data.from_placeholder(
            raw_data[-1],
            data.get_shape_out(),
            data.batch_size,
            sequence=False,
        )


    def make_sequence(self, max_length):
        return Data(self.dim,
                    datatype=self.datatype,
                    name=self.name,
                    sequence=True,
                    max_length=max_length,
                    batch_size=self.batch_size)

    def _forward(self, X):
        return X

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
