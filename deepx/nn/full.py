from .. import backend as T

from ..core import ShapedLayer, Data, Shape

class Linear(ShapedLayer):

    def initialize(self):
        if not self.is_elementwise():
            dim_in = self.get_dim_in()
            dim_out = self.get_dim_out()
            self.init_parameter('W', (dim_in, dim_out))
            self.init_parameter('b', dim_out)

    def activate(self, X):
        if self.is_elementwise():
            raise Exception("No identity nodes")
        return X

    def _forward(self, X, *args, **kwargs):
        if self.is_elementwise():
            return self.activate(X, *args)
        W, b = self.parameters['W'], self.parameters['b']
        return self.activate(T.dot(X, W) + b, *args)

class Maxout(Linear):


    def __init__(self, shape_in=None, shape_out=None, k=4, **kwargs):
        super(Maxout, self).__init__(shape_in=shape_in,
                                     shape_out=shape_out,
                                     **kwargs)
        self.k = k

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.get_shape_in(), self.get_shape_out())

    def initialize(self):
        dim_in = self.get_dim_in()
        dim_out = self.get_dim_out()
        self.init_parameter('W', (self.k, dim_in, dim_out))
        self.init_parameter('b', (self.k, dim_out))

    def activate(self, X):
        return T.max(X, axis=1)

class Softmax(Linear):

    def __init__(self, shape_in=None, shape_out=None, temp=1.0, temperature_parameter=False,
                 **kwargs):
        super(Softmax, self).__init__(shape_in=shape_in,
                                     shape_out=shape_out,
                                     **kwargs)
        self.T = temp
        self.temperature_parameter = temperature_parameter
        if self.temperature_parameter:
            self.T = Data(Shape(None), placeholder=T.placeholder(ndim=0, name='temperature'))

    def get_inputs(self):
        if self.temperature_parameter:
            return [self.T]
        return []

    def activate(self, X, *args):
        if self.temperature_parameter:
            return T.softmax(X, args[0])
        return T.softmax(X, self.T)

class Sigmoid(Linear):

    def activate(self, X):
        return T.sigmoid(X)

class Tanh(Linear):

    def activate(self, X):
        return T.tanh(X)

class Relu(Linear):

    def activate(self, X):
        return T.relu(X)

class Elu(Linear):

    def __init__(self, shape_in=None, shape_out=None, alpha=1.0, **kwargs):
        super(Elu, self).__init__(shape_in=shape_in,
                                     shape_out=shape_out,
                                     **kwargs)
        self.alpha = alpha

    def activate(self, X):
        return T.relu(X) + self.alpha * (T.exp((X - abs(X)) * 0.5) - 1)

class LeakyRelu(Linear):

    def __init__(self, shape_in=None, shape_out=None, alpha=1.0, **kwargs):
        super(LeakyRelu, self).__init__(shape_in=shape_in,
                                     shape_out=shape_out,
                                     **kwargs)
        self.alpha = alpha

    def activate(self, X):
        return T.relu(X, alpha=self.alpha)

class Tanlu(Linear):

    def initialize(self):
        self.init_parameter('alpha', self.get_shape_out()[0].get_dim(), value=0.5)

    def activate(self, X):
        alpha = self.get_parameter('alpha')
        constrained_alpha = T.clip(alpha, 0, 1)
        return constrained_alpha * T.tanh(X) + (1 - constrained_alpha) * T.relu(X)
