from deepx.backend import T
from deepx.core.op import Op

class Compose(Op):

    def __init__(self, left_op, right_op):
        self.left_op, self.right_op = left_op, right_op
        self.shape_inference()

    def forward(self, *inputs, **kwargs):
        params = kwargs.pop('params', None)
        if params is not None:
            left_params = params[0]
            right_params = params[1]
        else:
            left_params = right_params = None
        outputs = self.left_op.forward(*inputs, params=left_params)
        if self.left_op.get_shape_out() is None:
            self.left_op.set_shape_out([T.get_shape(output) for output in outputs])
        self.shape_inference()
        return self.right_op.forward(*outputs, params=right_params)

    @property
    def parameters(self):
        return self.left_op.parameters, self.right_op.parameters

    def get_parameters(self):
        return self.left_op.get_parameters() + self.right_op.get_parameters()

    def get_shape_in(self):
        return self.left_op.get_shape_in()

    def get_shape_out(self):
        return self.right_op.get_shape_out()

    def set_shape_in(self, shape_in):
        return self.left_op.set_shape_in(shape_in)

    def set_shape_out(self, shape_out):
        return self.right_op.set_shape_out(shape_out)

    def shape_inference(self):
        self.left_op.shape_inference()
        self.right_op.shape_inference()

        left_shape_out = self.left_op.get_shape_out()
        right_shape_in = self.right_op.get_shape_in()

        if left_shape_out is None:
            self.left_op.set_shape_out(right_shape_in)
            self.left_op.shape_inference()
        elif right_shape_in is None:
            self.right_op.set_shape_in(left_shape_out)
            self.right_op.shape_inference()
        else:
            if left_shape_out != right_shape_in:
                raise Exception("Shape inference error")

    def is_initialized(self):
        return self.left_op.is_initialized() and self.right_op.is_initialized()

    def __repr__(self):
        return "{left} >> {right}".format(
            left=repr(self.left_op),
            right=repr(self.right_op),
        )
