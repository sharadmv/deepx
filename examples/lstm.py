import numpy as np
from deepx.nn import *
from deepx.rnn import *
from deepx.optimize import rmsprop, cross_entropy
from deepx.util import *

if __name__ == "__main__":
    lstm = (Image('X', (1, 28, 28), 10) >> Conv((10, 2, 2)) >> Conv((20, 2, 2)) >> Flatten() >> LSTM(100))(10)[-1] >> Softmax(10)
    model = lstm | (predict, cross_entropy, rmsprop)
