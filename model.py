from keras.models import Sequential

from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D

def make_cnn():
    model = Sequential()

    model.add(Conv2D(96, (7, 7), padding = "same"))

    model.add(Conv2D(256, (5, 5), padding = "same"))

    model.add(Conv2D(384, (3, 3), padding = "same"))

    model.add(Dense(512))
    model.add(Dense(512))

    return model