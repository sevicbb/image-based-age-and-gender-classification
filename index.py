from data.loader import frontal_data_loader

from sklearn.metrics import confusion_matrix
from model import make_age_cnn, make_gender_cnn

import matplotlib.pyplot as plt

image_dims = (256, 256, 3)

def plot_confusion_matrix(matrix, title='Confusion matrix', cmap = plt.cm.Blues):
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # plt.show()

def main():
    data = frontal_data_loader()
    print(len(data))

    age_cnn = make_age_cnn(input_shape = image_dims)
    print(age_cnn)

    gender_cnn = make_gender_cnn(input_shape = image_dims)
    print(gender_cnn)

    conf_matrix = confusion_matrix([], [], labels=[0,1])	 
    plot_confusion_matrix(conf_matrix)

if __name__ == "__main__":
    main()