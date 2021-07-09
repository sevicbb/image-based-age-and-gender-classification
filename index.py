from model import make_model

from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from constants.index import age_list, gender_list
from data.loader import frontal_data_loader

import numpy as np
import matplotlib.pyplot as plt

image_dims = (256, 256, 3)

EPOCHS = 100
INIT_LR = 1e-3

def plot_confusion_matrix(matrix, title='Confusion matrix', cmap = plt.cm.Blues):
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # plt.show()

def create_model(classes):
    model = make_model(classes = classes, input_shape = image_dims)

    opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

    return model


def main():
    data = frontal_data_loader()
    print(len(data))

    age_train_labels = np.array(age_list)
    gender_train_labels = np.array(gender_list)

    age_label_binarizer = LabelBinarizer()
    age_label_binarizer.fit_transform(age_train_labels)

    gender_label_binarizer = LabelBinarizer()
    gender_label_binarizer.fit_transform(gender_train_labels)

    age_model = create_model(classes = len(age_label_binarizer.classes_))
    gender_model = create_model(classes = len(gender_label_binarizer.classes_))

    print(age_model)
    print(gender_model)

    conf_matrix = confusion_matrix([], [], labels=[0,1])	 
    plot_confusion_matrix(conf_matrix)

if __name__ == "__main__":
    main()