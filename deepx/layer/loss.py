import theano.tensor as T

from mixin import Mixin
from util import create_tensor

class Loss(Mixin):

    name = 'loss'

    def setup(self, model):
        self.y = create_tensor(model.arch.get_activation().ndim, name='y')
        super(Loss, self).setup(model)

    def get_inputs(self):
        return [i.get_data() for i in self.arch.get_inputs()] + [self.y]

    def get_result(self):
        return self.loss(self.arch.get_activation().get_data(), self.y)

    def loss(self, ypred, y):
        raise NotImplementedError

class cross_entropy(Loss):

    def loss(self, ypred, y):
        return T.nnet.categorical_crossentropy(ypred, y).mean()
