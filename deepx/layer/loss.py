import theano.tensor as T

from model import Data
from mixin import Mixin

class Loss(Mixin):

    name = 'loss'

    def get_aux_vars(self):
        if 'sequence' in self.model.mixins:
            return [T.tensor3('y')]
        return [T.matrix('y')]

    def mix(model, self, X, *args):
        if 'sequence' in self.model.mixins:
            activations = self.model.mixins['sequence'].mix(X, *args[:-1])
        else:
            data = Data(X, self.layer_vars)
            activations = (data > self.model).get_data()
        return self.mixin(activations, *args)

    def mixin(self, ypred, *args):
        y = args[-1]
        if ypred.ndim == 3:
            S, N, V = y.shape
            y = y.reshape((S * N, V))
            ypred = ypred.reshape((S * N, V))
        return self.loss(ypred, y)

    def __call__(self, *args, **kwargs):
        return self

class CrossEntropy(Loss):

    def loss(self, y_pred, y):
        return T.nnet.categorical_crossentropy(y_pred, y).mean()

cross_entropy = CrossEntropy()
