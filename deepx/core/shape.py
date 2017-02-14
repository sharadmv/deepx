from .. import T

class Shape(object):

    def __init__(self, shape,
                 batch=False,
                 sequence=False,
                 dtype=T.floatx()):
        if not (isinstance(shape, list) or isinstance(shape, tuple)):
            raise Exception("bad shape type")
        self.shape = list(shape)
        self.batch = batch
        self.sequence = sequence
        self.dtype = dtype

    def unify(self, other):
        if other.shape != self.shape:
            raise Exception("shape mismatch")
        return self.copy(
            sequence=other.sequence,
        )


    def is_none(self):
        return self.shape is None

    def get_shape(self):
        return self.shape

    def get_dim(self):
        dim = self.get_shape()
        if self.sequence:
            dim = dim[1:]
        if self.batch:
            dim = dim[1:]
        return dim

    def is_sequence(self):
        return self.sequence

    def get_dtype(self):
        return self.dtype

    def create_placeholder(self, name=None):
        if self.is_none():
            raise TypeError("Cannot create placeholder for Shape(None)")
        return T.placeholder(shape=self.shape, name=name, dtype=self.dtype)

    @classmethod
    def concatenate(self, shape_list):
        sequence = any(s.is_sequence() for s in shape_list)
        axis_dim = sum(s.get_dim()[-1] for s in shape_list)
        final_dim = shape_list[0].get_dim()[:-1] + (axis_dim,)
        final_shape = shape_list[0].copy(sequence=sequence,
                                     dim=final_dim)
        return final_shape

    def __eq__(self, other):
        if self.shape != other.shape:
            return False
        if self.sequence != other.sequence:
            return False
        if self.dtype != other.dtype:
            return False
        return True

    def __str__(self):
        return "{shape}".format(
             shape=self.shape
        )

    def __repr__(self):
        return "<Shape {shape}>".format(
            shape=self.shape
        )


    def copy(self,
             shape=None,
             batch=None,
             sequence=None,
             dtype=None
             ):
        if shape is None:
            shape = self.shape
        return Shape(shape,
                     batch=batch if batch is not None else self.batch,
                     sequence=sequence if sequence is not None else self.sequence,
                     dtype=dtype or self.dtype)
