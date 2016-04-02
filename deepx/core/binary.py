from .node import Node

class BinaryOpNode(Node):

    def __init__(self, left, right):
        super(BinaryOpNode, self).__init__()
        self.left = left
        self.right = right

    def get_inputs(self):
        return self.left.get_inputs(), self.right.get_inputs()

    def get_outputs(self, **kwargs):
        left_input, right_input = self.left.get_inputs(), self.right.get_inputs()
        return self.forward(left_input, right_input, **kwargs)

    def has_parameters(self):
        return self.left.has_parameters() or self.right.has_parameters()

    def set_shape_in(self, shape_in):
        raise NotImplementedError

    def set_shape_out(self, shape_out):
        raise NotImplementedError

    def get_shape_in(self):
        raise NotImplementedError

    def get_shape_out(self):
        raise NotImplementedError

    def is_configured(self):
        return self.left.is_configured() and self.right.is_configured()

    def get_parameters(self):
        params = []
        if self.left.has_parameters():
            params.extend(self.left.get_parameters())
        if self.right.has_parameters():
            params.extend(self.right.get_parameters())
        return params

class Chain(BinaryOpNode):

    def __init__(self, left, right):
        super(Chain, self).__init__(left, right)
        self.infer_shape()

    def get_network_inputs(self):
        inputs = []
        dups = set()
        for input in (self.left.get_network_inputs() + self.right.get_network_inputs()):
            if input not in dups:
                inputs.append(input)
                dups.add(input)
        return inputs

    def is_input(self):
        return self.left.is_input()

    def get_inputs(self):
        return self.left.get_inputs()

    def get_outputs(self, **kwargs):
        return self.forward(*self.get_inputs(), **kwargs)

    def forward(self, *left_inputs, **kwargs):
        left_outputs = self.left.forward(*left_inputs, **kwargs)
        right_outputs = self.right.forward(*left_outputs, **kwargs)
        return right_outputs

    def set_shape_in(self, shape_in):
        self.left.set_shape_in(shape_in)
        self.infer_shape()

    def set_shape_out(self, shape_out):
        self.right.set_shape_out(shape_out)

    def get_shape_in(self):
        return self.left.get_shape_in()

    def get_shape_out(self):
        return self.right.get_shape_out()

    def infer_shape(self):
        self.left.infer_shape()
        left_out = self.left.get_shape_out()
        self.right.set_shape_in(left_out)
        self.right.infer_shape()

    def __str__(self):
        return "%s >> %s" % (self.left, self.right)
