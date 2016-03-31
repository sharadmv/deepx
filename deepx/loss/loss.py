from .. import backend as T

class Loss(object):

    def __init__(self, model):
        self.model = model

        self._calc_loss = None
        self.updates = []
        self._grads = None

    def get_inputs(self):
        inputs = self.model.get_formatted_input()
        return inputs

    def get_updates(self):
        return self.model.get_updates() + self.updates

    def get_activation(self, **kwargs):
        return self.model.get_activation(**kwargs)

    def get_final_input(self):
        return self.get_inputs()

    def compute_loss(self, y, **kwargs):
        return self.loss(self.get_activation(use_dropout=True, **kwargs), y)

    def loss(self, y_pred, y):
        if y_pred.is_sequence():
            return T.mean(self.sequence_loss(y_pred, y))
        return self._loss(y_pred.get_data(), y)

    def sequence_loss(self, y_pred, y):
        def step(y_pred_i, y_i):
            return self._loss(y_pred_i, y_i)
        output, self.updates = T.scan(step, [y_pred.get_data(), y])
        return output

    def get_parameters(self):
        return self.model.get_parameters()


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

    def __init__(self, left, right):
        self.left, self.right = left, right
        self._calc_loss = None

    def get_activation(self, **kwargs):
        return self.left.get_activation(**kwargs)

    def get_updates(self):
        updates = self.left.get_updates()
        if isinstance(self.right, Loss):
            for update in self.right.get_updates():
                if update not in updates:
                    updates.append(update)
        return updates

    def get_inputs(self):
        inputs = self.left.get_inputs()
        if isinstance(self.right, Loss):
            for update in self.right.get_inputs():
                if update not in inputs:
                    inputs.append(update)
        return inputs

    def get_parameters(self):
        parameters = self.left.get_parameters()
        if isinstance(self.right, Loss):
            for parameter in self.right.get_parameters():
                if parameter not in parameters:
                    parameters.append(parameter)
        return parameters

    def compute_loss(self, y, **kwargs):
        left = self.left.compute_loss(y, **kwargs)
        if isinstance(self.right, Loss):
            right = self.right.compute_loss(y, **kwargs)
        else:
            right = self.right
        return self.op(left, right)

    def op(self, x, y):
        raise NotImplementedError

    def __str__(self):
        return "(%s %s %s)" % (self.left, self.op_str, self.right)

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
