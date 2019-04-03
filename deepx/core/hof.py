from abc import abstractmethod

from deepx.core.op import Op

class Unary(Op):

    def __init__(self, op):
        super(Unary, self).__init__()
        self.op = op

    def forward(self, *inputs):
        outputs = self.op.forward(*inputs)
        return self.combinator(inputs, outputs)

    def shape_inference(self):
        self.op.shape_inference()

    def get_shape_in(self):
        return self.op.get_shape_in()

    def get_shape_out(self):
        return self.op.get_shape_out()

    def set_shape_in(self, shape_in):
        self.op.set_shape_in(shape_in)

    def set_shape_out(self, shape_out):
        self.op.set_shape_out(shape_out)

    def is_initialized(self):
        return self.op.is_initialized()

    def get_parameters(self):
        return self.op.get_parameters()

    @abstractmethod
    def combinator(self, a, b):
        pass

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            repr(self.op)
        )

class Binary(Op):

    def __init__(self, left_op, right_op):
        super(Binary, self).__init__()
        self.left_op, self.right_op = left_op, right_op

    def forward(self, *inputs):
        left_out, right_out = (
            self.left_op.forward(*inputs),
            self.right_op.forward(*inputs)
        )
        return [self.combinator(a, b) for a, b in zip(left_out, right_out)]

    def shape_inference(self):
        self.left_op.shape_inference()
        self.right_op.shape_inference()

    def get_shape_in(self):
        return self.left_op.get_shape_in()

    def get_shape_out(self):
        return self.left_op.get_shape_out()

    def set_shape_in(self, shape_in):
        self.left_op.set_shape_in(shape_in)
        self.right_op.set_shape_in(shape_in)

    def set_shape_out(self, shape_out):
        self.left_op.set_shape_out(shape_out)
        self.right_op.set_shape_out(shape_out)

    def is_initialized(self):
        return self.left_op.is_initialized() and self.right_op.is_initialized()

    def get_parameters(self):
        return (self.left_op.get_parameters(), self.right_op.get_parameters())

    @abstractmethod
    def combinator(self, a, b):
        pass
