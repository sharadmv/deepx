import theano.tensor as T

from mixin import Mixin

class Loss(Mixin):

    name = 'loss'

    def get_aux_vars(self):
        return [T.matrix('y')]

    def mixin(self, activations, *args):
        return self.loss(activations[-1].get_data(), *args)

class CrossEntropy(Loss):

    def loss(self, y_pred, y):
        return T.nnet.categorical_crossentropy(y_pred, y).mean()

cross_entropy = CrossEntropy()
