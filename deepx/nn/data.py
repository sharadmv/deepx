from .. import backend as T

from ..node import Data

class Primitive(Data):

    def __init__(self, shape=0, batch_size=None, name=None):
        self.batch_size = batch_size
        self.name = name
        super(Primitive, self).__init__(self.get_var(name, shape), shape)

    def __str__(self):
        return "%s<%s>" % (self.__class__.__name__,
                           self.shape_out)

class Scalar(Primitive):

    def get_var(self, name, shape):
        return T.placeholder((self.batch_size,), name=name)

class Vector(Primitive):

    def get_var(self, name, shape):
        return T.placeholder((self.batch_size, shape), name=name)

class Matrix(Primitive):

    def get_var(self, name, shape):
        return T.placeholder((self.batch_size,) + shape, name=name)

class Image(Primitive):

    def get_var(self, name, shape):
        return T.placeholder((self.batch_size,) + shape, name=name)
