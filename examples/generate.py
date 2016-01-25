import numpy as np

from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    lstm = Vector(10, 10) >> Repeat(LSTM(100), 2) >> Softmax(10)
    discriminator = (LSTM(100) >> Softmax(2)).freeze()
    gen = Generate(lstm, 100) >> discriminator

    wat = RMSProp(gen, CrossEntropy())
