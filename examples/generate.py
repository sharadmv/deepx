import numpy as np

from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    lstm = Vector(100) >> Repeat(LSTM(1024), 2) >> Softmax(100)
    gen = Generate(lstm, 100)

    gen2 = Sequence(Vector(100, 1)) >> Repeat(LSTM(1024, stateful=True), 2) >> Softmax(100)

    # wat = RMSProp(gen, CrossEntropy())
