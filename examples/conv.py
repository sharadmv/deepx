from deepx.node import Image, predict, cross_entropy
from deepx.conv import Conv, Flatten
from deepx.nn import Softmax, Tanh
from deepx.optimize import rmsprop

if __name__ == "__main__":
    conv_net = Image('X', (1, 28, 28)) >> Conv((10, 2, 2)) >> Conv((20, 2, 2)) >> Flatten() >> Tanh(128) >> Softmax(10) | (predict, rmsprop, cross_entropy)
