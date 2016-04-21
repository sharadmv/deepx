import numpy as np

from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    lstm = Sequence(Vector(10, batch_size=2), 5) >> LSTM(2, stateful=True)
