from abc import abstractmethod
import six

from .. import T
from ..core import Node

__all__ = ['Identity', 'Add']

class SimpleOperator(Node):

    def __init__(self):
        super(SimpleOperator, self).__init__()
        self.dim_in = None
        self.dim_out = None

    def infer_shape(self, input_shapes):
        if len(input_shapes) == 0 or input_shapes[0] == None:
            return
        if isinstance(input_shapes, tuple):
            self.dim_in = input_shapes[0]
            self.dim_out = input_shapes[0]
        else:
            self.dim_in = input_shapes
            self.dim_out = input_shapes

    def is_initialized(self):
        return True

    def initialize(self):
        pass

    def get_dim_in(self):
        return self.dim_in

    def get_dim_out(self):
        return self.dim_out

class Identity(SimpleOperator):

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args

    def __repr__(self):
        return 'Identity(%s, %s)' % (self.get_dim_in(), self.get_dim_out())

# class Concatenate(SimpleOperator):

    # def forward(self, *inputs, **kwargs):
        # return T.concat(inputs, -1)

    # def get_num_outputs(self):
        # return 1

    # def infer_shape(self, *shapes_in):
        # if self.shapes_in is not None:
            # self.shapes_out = [Shape.concatenate(self.shapes_in)]

    # def __repr__(self):
        # return "Concatenate()"

    # def __str__(self):
        # return repr(self)


# class Sum(SimpleOperator):

    # def get_outputs(self, input, **kwargs):
        # raw_input = input.get_placeholder()
        # raw_output = T.sum(raw_input, axis=-1)
        # return [Data(self.get_shapes_out()[0], placeholder=raw_output)]

    # def get_num_inputs(self):
        # return 1

    # def get_num_outputs(self):
        # return 1

    # def infer_shape(self):
        # if self.shapes_in is not None:
            # shape_in = self.shapes_in[0]
            # self.shapes_out = [shape_in.copy(
                # dim=shape_in.get_dim()[:-1]
            # )]

    # def __repr__(self):
        # return "Sum()"

    # def __str__(self):
        # return repr(self)


# class Max(SimpleOperator):

    # def get_outputs(self, input, **kwargs):
        # raw_input = input.get_placeholder()
        # raw_output = T.max(raw_input, axis=-1)
        # return [Data(self.get_shapes_out()[0], placeholder=raw_output)]

    # def get_num_inputs(self):
        # return 1

    # def get_num_outputs(self):
        # return 1

    # def infer_shape(self):
        # if self.shapes_in is not None:
            # shape_in = self.shapes_in[0]
            # self.shapes_out = [shape_in.copy(
                # dim=shape_in.get_dim()[:-1]
            # )]

    # def __repr__(self):
        # return "Max()"

    # def __str__(self):
        # return repr(self)

# class UnaryOperator(SimpleOperator):

    # def forward(self, X):
        # return [self.op(X)]

    # @abstractmethod
    # def op(self, X):
        # pass

    # def infer_shape(self):
        # shapes_in = self.get_shapes_in()
        # if shapes_in is not None:
            # self.set_shapes_out([shapes_in[0].copy()])

    # def __str__(self):
        # return repr(self)

# class Pow(UnaryOperator):

    # def __init__(self, pow):
        # super(Pow, self).__init__()
        # self.pow = pow

    # def op(self, X):
        # return T.pow(X, self.pow)

    # def __repr__(self):
        # return "Pow(%u)" % self.pow

# class Exp(UnaryOperator):

    # def op(self, X):
        # return T.exp(X)

    # def __repr__(self):
        # return "Exp()"

class ArithmeticOperator(SimpleOperator):

    def forward(self, *inputs, **kwargs):
        raw_output = six.moves.reduce(self.op, inputs)
        return raw_output

    def __repr__(self):
        return "%s()" % self.op_name

    def __str__(self):
        return repr(self)

class Add(ArithmeticOperator):
    op_name = 'Add'

    @staticmethod
    def op(x, y):
        return x + y

# class Mul(ArithmeticOperator):
    # op_name = 'Mul'

    # @staticmethod
    # def op(x, y):
        # return x * y

# Prod = Mul


# class Sub(ArithmeticOperator):
    # op_name = 'Sub'

    # @staticmethod
    # def op(x, y):
        # return x - y

# class Div(ArithmeticOperator):
    # op_name = 'Div'

    # @staticmethod
    # def op(x, y):
        # return x / y

# class Index(SimpleOperator):

    # def __init__(self, index):
        # super(Index, self).__init__()
        # self.index = index

    # def get_outputs(self, input, **kwargs):
        # raw_input = input.get_placeholder()
        # if T.ndim(raw_input) == 2:
            # raw_output = raw_input[:, self.index]
        # if T.ndim(raw_input) == 3:
            # raw_output = raw_input[:, :, self.index]
        # return [Data(self.get_shapes_out()[0], placeholder=raw_output)]

    # def get_num_inputs(self):
        # return 1

    # def get_num_outputs(self):
        # return 1

    # def infer_shape(self):
        # if self.shapes_in is not None:
            # shape_in = self.shapes_in[0]
            # self.shapes_out = [shape_in.copy(
                # dim=shape_in.get_dim()[:-1]
            # )]

    # def __repr__(self):
        # return "Index()"

    # def __str__(self):
        # return repr(self)
