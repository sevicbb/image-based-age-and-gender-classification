from data.loader import data_loader, frontal_data_loader
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

def plot_confusion_matrix(matrix, title='Confusion matrix', cmap = plt.cm.Blues):
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    plt.show()

def main():
    data_loader()
    frontal_data_loader()

    conf_matrix = confusion_matrix([], [], labels=[0,1])	 
    plot_confusion_matrix(conf_matrix)

if __name__ == "__main__":
    main()