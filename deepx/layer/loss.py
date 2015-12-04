import theano.tensor as T

from model import Model

class Loss(object):

    def loss(self, y_pred, y):
        raise NotImplementedError

    def __ne__(self, layer):
        return Model(layer, self)

    def str(self):
        pass


class MSE(Loss):

    def loss(self, y_pred, y):
        return ((y - y_pred) ** 2).mean()

class CrossEntropy(Loss):

    def loss(self, y_pred, y):
        return T.nnet.categorical_crossentropy(y_pred, y).mean()
