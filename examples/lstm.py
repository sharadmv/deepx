import theano.tensor as T
import numpy as np

from deepx.optimize import rmsprop
from deepx.layer import *

if __name__ == "__main__":
    data_dim = 1

    X = Matrix('X')
    Y = Matrix('Y')

    model1 = X > Tanh(10, 20) >> Softmax(20, 10)
    model2 = Y > Tanh(10, 20) >> Softmax(20, 10)

    model3 = model1 + model2

    #model2 = LSTM(10, 50) >> LSTM(40, 10) >> Softmax(10, 20) | output

    #x = model1.output()
    #y = model1.output()

    #model.compile()
