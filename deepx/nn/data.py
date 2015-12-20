from .. import backend as T

from ..node import Data

class Primitive(Data):

    def __init__(self, name, shape):
        self.name = name
        super(Primitive, self).__init__(self.get_var(name, shape), shape)

    def __str__(self):
        return "%s<%s, %s>" % (self.__class__.__name__,
                               self.name, self.shape_out)

class Vector(Primitive):

    def get_var(self, name, shape):
        return T.placeholder((None, shape), name=name)

class Image(Primitive):

    def get_var(self, name, shape):
        return T.placeholder((None,) + shape, name=name)
