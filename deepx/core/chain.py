from .binary import BinaryOpNode

__all__ = ['Chain']

class Chain(BinaryOpNode):

    def __init__(self, left, right):
        super(Chain, self).__init__(left, right)
        if left.get_num_outputs() != right.get_num_inputs():
            raise TypeError("Cannot chain %s and %s" % (left, right))

    def get_outputs(self, *inputs, **kwargs):
        left_out = self.left.get_outputs(*inputs, **kwargs)
        right_out = self.right.get_outputs(*left_out, **kwargs)
        return right_out

    def get_shapes_in(self):
        return self.left.get_shapes_in()

    def get_shapes_out(self):
        return self.right.get_shapes_out()

    def set_shapes_in(self, shapes_in):
        self.left.set_shapes_in(shapes_in)

    def set_shapes_out(self, shapes_out):
        self.right.set_shapes_out(shapes_out)

    def get_num_inputs(self):
        return self.left.get_num_inputs()

    def get_num_outputs(self):
        return self.right.get_num_outputs()

    def infer_shape(self):
        self.left.infer_shape()
        self.right.infer_shape()
        if self.left.get_shapes_out() is not None and self.left.get_shapes_out() != self.right.get_shapes_in():
            self.right.set_shapes_in(self.left.get_shapes_out())
            self.right.infer_shape()

    def get_state(self, **kwargs):
        return (self.left.get_state(**kwargs), self.right.get_state(**kwargs))

    def set_state(self, state):
        left, right = state
        self.left.set_state(left)
        self.right.set_state(right)

    def __repr__(self):
        return "%s >> %s" % (self.left, self.right)

    def __str__(self):
        return repr(self)

