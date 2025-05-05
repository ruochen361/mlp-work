import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.iloc[:, 1:].values / 255.0
    y_train = train.iloc[:, 0].values
    X_test = test.iloc[:, 1:].values / 255.0
    y_test = test.iloc[:, 0].values

    # One-hot encode labels
    y_train_onehot = np.eye(10)[y_train]
    return X_train, y_train_onehot, X_test, y_test


def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(classes), range(classes))
    plt.yticks(np.arange(classes), range(classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()