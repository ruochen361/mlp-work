import numpy as np

# 损失函数
class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size

    @staticmethod
    def cross_entropy(y_true, y_pred, epsilon=1e-12):
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred))

    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]


loss_functions = {
    'mse': (Loss.mse, Loss.mse_derivative),
    'cross_entropy': (Loss.cross_entropy, Loss.cross_entropy_derivative)
}