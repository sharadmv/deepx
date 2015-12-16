import theano.tensor as T

from ..node import Mixin
from ..util import create_tensor

class Loss(Mixin):

    name = 'loss'

    def setup(self, model):
        super(Loss, self).setup(model)
        self.y = create_tensor(self.get_activation().ndim, 'y')

    def get_inputs(self):
        return super(Loss, self).get_inputs() + [self.y]

    def get_result(self):
        ypred = self.get_activation().get_data()
        y = self.y
        if ypred.ndim == 3:
            S, N, V = ypred.shape
            y = y.reshape((S * N, V))
            ypred = ypred.reshape((S * N, V))
        return self.loss(ypred, y)

    def loss(self, ypred, y):
        raise NotImplementedError

class cross_entropy(Loss):

    def loss(self, ypred, y):
        return T.nnet.categorical_crossentropy(ypred, y).mean()

class mse(Loss):

    def loss(self, ypred, y):
        return T.mean((ypred - y) ** 2)
