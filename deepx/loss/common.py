from .. import T

from .loss import Loss

class CrossEntropy(Loss):

    def loss(self, ypred, y):
        return T.mean(T.categorical_crossentropy(ypred, y))

class LogLoss(Loss):

    def loss(self, ypred, y):
        ypred = T.clip_by_value(ypred, T.epsilon(), 1 - T.epsilon())
        return -T.mean(y * T.log(ypred) + (1 - y) * T.log(1 - ypred))

class MSE(Loss):

    def loss(self, ypred, y):
        return T.mean(T.pow((ypred - y), 2))
