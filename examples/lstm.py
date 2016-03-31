import numpy as np

from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    lstm = Sequence(Vector(10)) >> Repeat(MaxoutLSTM(20) >> Dropout(0.1), 2) >> Maxout(100) >> Tanh() >> Softmax(10)
