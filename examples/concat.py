import numpy as np
from deepx.nn import Softmax, Tanh, predict, Matrix
from deepx.optimize import rmsprop, cross_entropy

if __name__ == "__main__":
    model1 = Matrix('X', 10) >> Tanh(30)
    model2 = Matrix('Y', 20) >> Tanh(40)


    model3 = (model1 + model2) >> Softmax(10) | predict
