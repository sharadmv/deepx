import numpy as np
from deepx.nn import *
from deepx.rnn import *

if __name__ == "__main__":
    lstm = Sequence(Vector(101), 10) >> Repeat(LSTM(20), 2) >> Softmax(2)
