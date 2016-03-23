from .. import backend as T

class Loss(object):

    def __init__(self, model):
        self.model = model
        self.ypred = self.model.get_activation(use_dropout=True)
        self.y = T.placeholder(shape=self.ypred.get_data())

        self._calc_loss = None

    def get_inputs(self):
        inputs = self.get_model().get_formatted_input()
        return inputs + [self.y]

    def batch_loss(self, *args):
        if self._calc_loss is None:
            self._calc_loss = T.function(self.get_inputs(), [self.get_loss()], updates=self.get_model().get_updates())
        return self._calc_loss(*args)

    def get_loss(self):
        return self.loss(self.ypred, self.y)

    def get_model(self):
        return self.model

    def loss(self, y_pred, y):
        if y_pred.is_sequence():
            return T.mean(self.sequence_loss(y_pred, y))
        return self._loss(y_pred.get_data(), y)

    def sequence_loss(self, y_pred, y):
        def step(ypred_i, y_i):
            return self._loss(ypred_i, y_i)
        output = T.scan(step, [y_pred.get_data(), y])
        return output

    def __mul__(self, x):
        return MulLoss(self, x)

    def __rmul__(self, x):
        return MulLoss(self, x)

    def __add__(self, x):
        return AddLoss(self, x)

    def __radd__(self, x):
        return AddLoss(self, x)

    def __sub__(self, x):
        return SubLoss(self, x)

    def __rsub__(self, x):
        return SubLoss(self, x)

    def __div__(self, x):
        return DivLoss(self, x)

    def __rdiv__(self, x):
        return DivLoss(self, x)

    def __str__(self):
        return self.__class__.__name__

class ArithmeticLoss(Loss):


    def __init__(self, loss, num):
        self.loss = loss
        self.num = num

        self._calc_loss = None

    def loss(self, *args):
        raise NotImplementedError

    def get_model(self):
        return self.loss.get_model()

    def get_inputs(self):
        return self.loss.get_inputs()

    def get_loss(self):
        return self.op(self.loss.get_loss(), self.num)

    def op(self, x, y):
        raise NotImplementedError

    def __str__(self):
        return "(%s %s %s)" % (self.loss, self.op_str, self.num)

class MulLoss(ArithmeticLoss):

    op_str = '*'

    def op(self, x, y):
        return x * y

class DivLoss(ArithmeticLoss):

    op_str = '/'

    def op(self, x, y):
        return x / y

class AddLoss(ArithmeticLoss):

    op_str = '+'

    def op(self, x, y):
        return x + y

class SubLoss(ArithmeticLoss):

    op_str = '-'

    def op(self, x, y):
        return x - y
