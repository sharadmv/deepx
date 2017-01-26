from .. import T
import tensorflow as tf

from .loss import Loss

class CrossEntropy(Loss):

    def loss(self, ypred, y):
        return T.mean(T.categorical_crossentropy(ypred, y))

class LogLoss(Loss):

    def __init__(self, *args, **kwargs):
        use_ones = kwargs.pop('use_ones', False)
        self.use_ones = use_ones
        super(LogLoss, self).__init__(*args, **kwargs)
        self.y = 1

    def loss(self, ypred, y):
        ypred = T.clip_by_value(ypred, T.epsilon(), 1 - T.epsilon())
        if self.use_ones:
            return -T.mean(T.log(ypred))
        return -T.mean(y * T.log(ypred) + (1 - y) * T.log(1 - ypred))

class SampledLogLoss(Loss):

    def __init__(self, one_ratio, *args, **kwargs):
        super(SampledLogLoss, self).__init__(*args, **kwargs)
        self.one_ratio = one_ratio

    def loss(self, ypred, y):
        one_ratio = tf.reducejsum(y) / tf.to_float(tf.size(y))
        noise = T.random_binomial(T.shape(y), p=one_ratio)
        cost = y * T.log(ypred) + (1 - y) * T.log(1 - ypred)
        mask = y + noise > 0
        mask.set_shape(ypred.get_shape())
        cost = tf.boolean_mask(cost, mask)
        return -T.mean(cost)

class MSE(Loss):

    def loss(self, ypred, y):
        return T.mean(T.pow((ypred - y), 2))
