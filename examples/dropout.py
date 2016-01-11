from deepx.nn import *
from deepx.rnn import *

if __name__ == "__main__":

    layer = Tanh(200) >> Dropout(0.5)
    mlp = Vector(784) >> Repeat(layer, 2) >> Softmax(10)
