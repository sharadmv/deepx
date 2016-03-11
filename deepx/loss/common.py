from .. import backend as T

from .loss import Loss

class CrossEntropy(Loss):

    def _loss(self, ypred, y):
        return T.mean(T.categorical_crossentropy(ypred, y))

class LogLoss(Loss):

    def _loss(self, ypred, y):
        return - T.mean(y * T.log(ypred) + (1 - y) * T.log(1 - ypred))

class MSE(Loss):

    def _loss(self, ypred, y):
        return T.mean(T.pow((ypred - y), 2))
