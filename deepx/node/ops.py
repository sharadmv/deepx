from node import Node, CompositeNode, NestedNode
from .. import backend as T

class ArithmeticNode(CompositeNode):

    def __init__(self, left, right):
        super(ArithmeticNode, self).__init__(left, right)

        self.op = lambda x, y: x
        def calculate(x, y):
            x_d, y_d = x.get_data(), y.get_data()
            return x.next(self.op(x_d, y_d), self.left.get_shape_out())
        self._calculate = calculate

    def is_initialized(self):
        return self.left.is_initialized() and self.right.is_initialized()

    def recurrent_forward(self, X, **kwargs):
        raise NotImplementedError

    def step(self, X, state):
        raise NotImplementedError

    def infer_shape(self):
        self.left.infer_shape()
        self.right.infer_shape()

    def set_shape_in(self, shape_in):
        raise NotImplementedError

    def set_shape_out(self, shape_out):
        raise NotImplementedError

    def get_shape_in(self):
        return [self.left.get_shape_in(), self.right.get_shape_in()]

    def get_shape_out(self):
        return self.left.get_shape_out()

    def copy(self, **kwargs):
        raise NotImplementedError

    def tie(self, node):
        raise NotImplementedError

    def __str__(self):
        return "{left} {op_str} {right}".format(
            left=self.left,
            right=self.right,
            op_str=self.op_str
        )

    def _infer(self, shape_in):
        return shape_in[0]

    def forward(self, X, **kwargs):
        left_input, right_input = X
        left, right = self.left.forward(left_input, **kwargs), self.right.forward(right_input, **kwargs)
        return self._calculate(left, right)

    def get_input(self):
        return [self.left.get_input(), self.right.get_input()]

    def get_formatted_input(self):
        inputs = self.left.get_formatted_input()
        for input in self.right.get_formatted_input():
            if input not in inputs:
                inputs.append(input)
        return inputs

class AddNode(ArithmeticNode):

    op_str = '+'

    def __init__(self, left, right):
        super(AddNode, self).__init__(left, right)
        self.op = lambda x, y: x + y

class SubtractNode(ArithmeticNode):

    op_str = '-'

    def __init__(self, left, right):
        super(SubtractNode, self).__init__(left, right)
        self.op = lambda x, y: x - y

class ArithmeticScalar(NestedNode):

    def __init__(self, node, num):
        super(ArithmeticScalar, self).__init__(node)
        self.op = lambda x, y: x
        self.num = num
        def calculate(x):
            x_d = x.get_data()
            return x.next(self.op(x_d, self.num), self.node.get_shape_out())
        self._calculate = calculate


    def _infer(self, shape_in):
        return shape_in

    def forward(self, X, **kwargs):
        X = self.node.forward(X, **kwargs)
        return self._calculate(X)

    def __str__(self):
        return "{left} {op_str} {right}".format(
            left=self.num,
            right=self.node,
            op_str=self.op_str
        )

class AddScalar(ArithmeticScalar):

    op_str = '+'

    def __init__(self, node, num):
        super(AddScalar, self).__init__(node, num)
        self.op = lambda x, y: x + y
        self.num = num

class SubtractScalar(ArithmeticScalar):

    op_str = '-'

    def __init__(self, node, num):
        super(SubtractScalar, self).__init__(node, num)
        self.op = lambda x, y: x - y

class MultiplyScalar(ArithmeticScalar):

    op_str = '*'

    def __init__(self, node, num):
        super(MultiplyScalar, self).__init__(node, num)
        self.op = lambda x, y: x * y
