from .. import T
from .loss import Loss

class SequentialLoss(Loss):

    def __init__(self, loss):
        self.inner_loss = loss

    def _loss(self, *args):
        return self.inner_loss._loss(*args)

    def loss(self, ypred, y):
        raise NotImplementedError

class ConvexSequentialLoss(SequentialLoss):

    def __init__(self, loss, a):
        super(ConvexSequentialLoss, self).__init__(loss)
        assert 0 <= a <= 1, "Invalid convex combination"
        self.a = a

    def loss(self, ypred, y):
        sequence_loss = self.sequence_loss(ypred, y)
        length = T.shape(sequence_loss)[0]
        first_loss = T.mean(sequence_loss[:length - 1])
        last_loss = sequence_loss[length - 1]
        return self.a * first_loss + (1 - self.a) * last_loss

class LinearSequentialLoss(SequentialLoss):

    def loss(self, ypred, y):
        sequence_loss = self.sequence_loss(ypred, y)
        length = T.shape(sequence_loss)[0]
        interpolation = T.interpolate(length)
        return T.sum(interpolation * sequence_loss)
