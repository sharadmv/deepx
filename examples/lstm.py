import numpy as np
from deepx.nn import *
from deepx.rnn import *
from deepx.optimize import rmsprop, cross_entropy

if __name__ == "__main__":
    lstm = (Image((1, 28, 28), 100) >> Conv((10, 2, 2)) >> Conv((20, 2, 2)) >> Flatten() >> LSTM(100))(100)[-1] >> Softmax(10)
    model = lstm | (predict, cross_entropy, rmsprop)
