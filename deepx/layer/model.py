import theano
import theano.tensor as T
from theanify import theanify, Theanifiable

class Data(object):

    def __init__(self, X):
        self.X = X

    def __gt__(self, layer):
        return layer.forward(self)

class SequenceData(object):

    def __init__(self, n_dim, X, H=None):
        self.n_dim = n_dim
        self.X = X
        self.H = H

    def __gt__(self, layer):
        H = self.H
        if H is None:
            H = layer.get_defaults()
        return layer.step(self.X, H)

class Model(Theanifiable):

    def __init__(self, arch, loss):
        super(Model, self).__init__()
        self.arch = arch
        self.cost_function = loss
        self.compile_method('forward')

    def __lt__(self, X):
        return self.forward(X)

    @theanify(T.matrix('X'))
    def forward(self, X):
        return self.arch.forward(X)

    @theanify(T.matrix('X'), T.matrix('y'))
    def cost(self, X, y):
        y_pred = self.arch.forward(X)
        return self.cost_function.loss(y_pred, y)

    def get_parameters(self):
        return self.arch.get_parameters()

    def __str__(self):
        return "(%s) <> %s" % (str(self.arch), str(self.cost_function))

class SequenceModel(Model):

    def __init__(self, arch, loss):
        super(Model, self).__init__()
        self.arch = arch
        self.cost_function = loss
        self.defaults = self.arch.get_defaults()
        self.num_defaults = len(self.defaults)
        self.forward_args = [T.tensor3('X')]  + [p for a in self.defaults for p in a]
        self.compile_method('forward', args=self.forward_args)

    def get_initial(self, batch_size):
        initial = self.arch.get_initial(batch_size)
        return [p for a in initial for p in a]

    def __lt__(self, X):
        return self.forward(X)

    @theanify(returns_updates=True)
    def forward(self, X, *args):

        def step(input, *args):
            formatted_args = zip(*[iter(args)] * (len(args) / self.num_defaults))
            X, H = self.arch.step(input, formatted_args)
            return X, H
        wat, updates = theano.scan(step,
                                   sequences=[X],
                                   outputs_info=args)
        return wat, updates

    @theanify(T.tensor3('X'), T.tensor3('y'))
    def cost(self, X, y):
        return X, H

    def get_parameters(self):
        return self.arch.get_parameters()

    def __str__(self):
        return "(%s) <> %s" % (str(self.arch), str(self.cost_function))

