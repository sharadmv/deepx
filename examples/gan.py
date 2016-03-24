import numpy as np

from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    gen = Generate(Vector(10, 10, name='X') >> Repeat(LSTM(20), 2) >> Softmax(10), 10)

    disc = Repeat(LSTM(20), 2) >> Softmax(2)
    panel = [gen >> (disc.copy()) for _ in xrange(1)]
    losses = [CrossEntropy(l) for l in panel]

    final_loss = sum(losses) / 5.0

    rmsprop = RMSProp(final_loss)
