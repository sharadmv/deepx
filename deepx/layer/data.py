import theano.tensor as T

class Data(object):

    def __init__(self, data):
        super(Data, self).__init__()
        self.data = data

    @property
    def ndim(self):
        return self.data.ndim

    def __gt__(self, node):
        node.add_input(self)
        node.propagate()
        return node

    def get_data(self):
        return self.data

    def __str__(self):
        return "%s<%s>" % (self.__class__.__name__, self.data)

    def __repr__(self):
        return str(self)

class Primitive(Data):

    def __init__(self, name):
        super(Primitive, self).__init__(self.get_var(name))

class Vector(Primitive):

    def get_var(self, name):
        return T.vector(name)

class Matrix(Primitive):

    def get_var(self, name):
        return T.matrix(name)

class Sequence(Primitive):

    def get_var(self, name):
        return T.tensor3(name)
