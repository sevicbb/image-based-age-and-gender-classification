from data.loader import data_loader, frontal_data_loader
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

def plot_confusion_matrix(matrix, title='Confusion matrix', cmap = plt.cm.Blues):
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

def main():
    data_loader()
    frontal_data_loader()

    # plot_confusion_matrix(None)

if __name__ == "__main__":
    main()