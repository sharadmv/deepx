import numpy as np

from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    gen = Generate(Vector(100) >> Repeat(LSTM(1024), 2) >> Softmax(100), 100)
