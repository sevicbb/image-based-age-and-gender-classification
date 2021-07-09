from keras.models import Sequential

from keras.layers.core import Dense
from keras.layers.core import Activation

from keras.layers.convolutional import Conv2D

def make_model(input_shape):
    model = Sequential()

    model.add(Conv2D(96, (7, 7), padding = "same", input_shape = input_shape))

    model.add(Conv2D(256, (5, 5), padding = "same"))

    model.add(Conv2D(384, (3, 3), padding = "same"))

    model.add(Dense(512))
    model.add(Dense(512))

    model.add(Activation("softmax"))

    return model
