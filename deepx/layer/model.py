import theano.tensor as T
from theanify import theanify, Theanifiable

class Data(object):

    def __init__(self, n_dim, X):
        self.n_dim = n_dim
        self.X = X

    def __gt__(self, layer):
        return layer.chain(self)


class Model(Theanifiable):

    def __init__(self, arch, loss):
        super(Model, self).__init__()
        self.arch = arch
        self.cost_function = loss

    @theanify(T.matrix('X'))
    def forward(self, X):
        return self.arch.forward(X)

    @theanify(T.matrix('X'), T.matrix('y'))
    def cost(self, X, y):
        y_pred = self.arch.forward(X)
        return self.cost_function.loss(y_pred, y)

    def get_parameters(self):
        return self.arch.get_parameters()
