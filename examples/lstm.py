import numpy as np

from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    lstm = Sequence(Vector(1, 10), 10) >> Repeat(LSTM(1, stateful=True) >> Dropout(0.1), 2) >> Softmax(10)
