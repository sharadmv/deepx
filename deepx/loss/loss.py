class Loss(object):

    def loss(self, y_pred, y):
        raise NotImplementedError

    def __add__(self, other):
        return CompositeLoss(self, other)

class CompositeLoss(Loss):

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def loss(self, *args):
        return self.left.loss(*args) + self.right.loss(*args)
