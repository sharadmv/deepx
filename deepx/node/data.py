import theano.tensor as T

from node import Data

class Primitive(Data):

    def __init__(self, name, shape):
        self.name = name
        super(Primitive, self).__init__(self.get_var(name), shape)

    def to_str(self):
        return "%s<%s, %s>" % (self.__class__.__name__,
                               self.name, self.shape_out)

class Vector(Primitive):

    def get_var(self, name):
        return T.vector(name)

class Matrix(Primitive):

    def get_var(self, name):
        return T.matrix(name)

class Sequence(Primitive):

    def get_var(self, name):
        return T.tensor3(name)

class Image(Primitive):

    def get_var(self, name):
        return T.tensor4(name)
