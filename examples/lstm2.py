import numpy as np
from deepx.nn import *
from deepx.rnn import *

if __name__ == "__main__":
    discriminator = Last(Sequence(Vector(101, 10), 10) >> Repeat(LSTM(1024), 2)) >> Softmax(2)
