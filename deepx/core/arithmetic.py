from abc import abstractmethod

from deepx.core.hof import Binary

__all__ = ["Add", "Sub", "Mul", "Div"]

class Add(Binary):

    def combinator(self, a, b):
        return a + b

    def __repr__(self):
        return "{} + {}".format(
            self.left_op,
            self.right_op
        )

class Sub(Binary):

    def combinator(self, a, b):
        return a - b

    def __repr__(self):
        return "{} - {}".format(
            self.left_op,
            self.right_op
        )

class Mul(Binary):

    def combinator(self, a, b):
        return a * b

    def __repr__(self):
        return "{} * {}".format(
            self.left_op,
            self.right_op
        )

class Div(Binary):

    def combinator(self, a, b):
        return a / b

    def __repr__(self):
        return "{} / {}".format(
            self.left_op,
            self.right_op
        )
