from .. import backend as T

class Loss(object):

    def loss(self, y_pred, y):
        if y_pred.is_sequence():
            return T.mean(self.sequence_loss(y_pred, y))
        return self._loss(y_pred.get_data(), y)

    def sequence_loss(self, y_pred, y):
        def step(ypred_i, y_i):
            return self._loss(ypred_i, y_i)
        output = T.scan(step, [y_pred.get_data(), y])
        return output

    def __add__(self, other):
        return CompositeLoss(self, other)

class CompositeLoss(Loss):

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def loss(self, *args):
        return self.left.loss(*args) + self.right.loss(*args)
