import matplotlib.pyplot as plt
import theano.tensor as T
import numpy as np

from deepx.layer import *
from deepx.optimize import rmsprop, sgd

if __name__ == "__main__":

    X = Matrix('X')
    Y = Matrix('Y')

    arch1 = X > Tanh(10, 20) >> Tanh(20, 30) >> Softmax(30, 10)
    arch2 = Y > Tanh(10, 30)
    arch3, model = (arch1 + arch2) >> Softmax(40, 10) | (predict, cross_entropy, rmsprop)
    arch3, model2 = (arch1 + arch2) >> Softmax(40, 10) | (cross_entropy, sgd)

    x = np.zeros((20, 10))
    y = np.zeros((20, 10))
