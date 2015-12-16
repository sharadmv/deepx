import theano.tensor as T

from ..node import Data
from ..util import create_tensor

class Primitive(Data):

    def __init__(self, name, shape):
        self.name = name
        super(Primitive, self).__init__(self.get_var(name), shape)

    def __str__(self):
        return "%s<%s, %s>" % (self.__class__.__name__,
                               self.name, self.shape_out)

class Vector(Primitive):

    def get_var(self, name):
        return T.vector(name)

class Matrix(Primitive):

    def get_var(self, name):
        return T.matrix(name)

class Image(Primitive):

    def get_var(self, name):
        return T.tensor4(name)

class Sequence(Data):

    def __init__(self, data_var):
        self.data_var = data_var
        self.sequence_dim = data_var.ndim

        self.data = create_tensor(self.sequence_dim + 1)
        self.shape_in = self.data_var.shape_in
        self.shape_out = self.data_var.shape_out

    def __str__(self):
        return "Sequence(%s)" % self.data_var

    def is_sequence(self):
        return True
