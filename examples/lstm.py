import numpy as np

from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    # lstm = Sequence(Image((1, 28, 28), 10), 10) >> Conv((10, 2, 2)) >> Conv((20, 2, 2)) >> Flatten() >> LSTM(100)
    # model = lstm >> Softmax(10)

    lstm = Sequence(Vector(1, 10)) >> Repeat(LSTM(1, stateful=True), 2) >> Softmax(10)

    # rmsprop = RMSProp(model, LinearSequentialLoss(CrossEntropy()))
    # rmsprop = RMSProp(lstm, CrossEntropy())
