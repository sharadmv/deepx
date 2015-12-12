from deepx.nn import Image, Softmax, Tanh, Flatten, Conv, predict
from deepx.optimize import rmsprop, cross_entropy

if __name__ == "__main__":
    conv_net = Image('X', (1, 28, 28)) >> Conv((10, 2, 2)) >> Conv((20, 2, 2)) >> Flatten() >> Tanh(128) >> Softmax(10) | (predict, rmsprop, cross_entropy)
