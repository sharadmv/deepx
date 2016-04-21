from .. import backend as T

class Shape(object):

    def __init__(self, dim,
                 batch_size=None,
                 sequence=False,
                 max_length=None,
                 dtype=T.floatx()):
        if isinstance(dim, int):
            self.dim = (dim,)
        else:
            self.dim = dim
        self.batch_size = batch_size
        self.sequence = sequence
        self.max_length = max_length
        self.dtype = dtype

    def get_dim(self):
        return self.dim

    def get_batch_size(self):
        return self.batch_size

    def is_sequence(self):
        return self.sequence

    def get_max_length(self):
        return self.max_length

    def get_dtype(self):
        return self.dtype

    def create_placeholder(self, name=None):
        if self.sequence:
            shape = [self.max_length, self.batch_size] + list(self.dim)
        else:
            shape = [self.batch_size] + list(self.dim)
        return T.placeholder(shape=shape, name=name, dtype=self.dtype)

    @classmethod
    def concatenate(self, shape_list):
        sequence = any(s.is_sequence() for s in shape_list)
        axis_dim = sum(s.get_dim()[-1] for s in shape_list)
        final_dim = shape_list[0].get_dim()[:-1] + (axis_dim,)
        final_shape = shape_list[0].copy(sequence=sequence,
                                     dim=final_dim)
        return final_shape

    def __eq__(self, other):
        if self.dim != other.dim:
            return False
        # if self.batch_size != other.batch_size:
            # return False
        if self.sequence != other.sequence:
            return False
        if self.dtype != other.dtype:
            return False
        return True

    def __str__(self):
        dim = self.dim
        if len(dim) == 1:
            dim = self.dim[0]
        if self.sequence:
            return "{dim}".format(
                dim=dim
            )
        return "{dim}".format(
             dim=dim
        )

    def __repr__(self):
        dim = self.dim
        if len(self.dim) == 1:
            dim = self.dim[0]
        if self.sequence:
            return "<Shape {dim}>".format(
                dim=dim
            )
        return "<Shape {dim}>".format(
            dim=dim
        )


    def copy(self,
             dim=None,
             batch_size=None,
             sequence=None,
             max_length=None,
             dtype=None
             ):
        if dim is None:
            dim = self.dim
        return Shape(dim,
                     batch_size=batch_size or self.batch_size,
                     sequence=sequence if sequence is not None else self.sequence,
                     max_length=max_length or self.max_length,
                     dtype=dtype or self.dtype)
