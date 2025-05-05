import numpy as np
from sklearn.metrics import confusion_matrix

from model import MLPModel
from optimizers import Adam
from utils import load_mnist, plot_confusion_matrix

# mnist数据集
X_train, y_train, X_test, y_test = load_mnist(
    "mnist_train.csv",
    "mnist_test.csv"
)

# Initialize model
model = MLPModel(
    layers=[784, 128, 64, 10],
    activations=['relu', 'relu', 'softmax'],
    task='classification',
    weight_init='he',
    reg='l2',
    reg_lambda=0.01
)

# Train
model.train(
    X_train,
    y_train,
    epochs=20,
    batch_size=128,
    optimizer=Adam(learning_rate=0.001),
    validation_data=(X_test, y_test),
    verbose=True
)

# Evaluate
test_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test.argmax(axis=1), test_pred, 10)
plot_confusion_matrix(conf_matrix, 10)
print(f"Test Accuracy: {np.mean(test_pred == y_test.argmax(axis=1)):.4f}")