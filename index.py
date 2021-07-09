from data.loader import frontal_data_loader

from sklearn.metrics import confusion_matrix
from model import make_model

from keras.optimizers import Adam

import matplotlib.pyplot as plt

image_dims = (256, 256, 3)

EPOCHS = 100
INIT_LR = 1e-3

def plot_confusion_matrix(matrix, title='Confusion matrix', cmap = plt.cm.Blues):
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # plt.show()

def create_model():
    model = make_model(input_shape = image_dims)

    opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

    return model


def main():
    data = frontal_data_loader()
    print(len(data))

    age_model = create_model()
    print(age_model)

    gender_model = create_model()
    print(gender_model)

    conf_matrix = confusion_matrix([], [], labels=[0,1])	 
    plot_confusion_matrix(conf_matrix)

if __name__ == "__main__":
    main()