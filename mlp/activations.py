import numpy as np

# 激活函数
class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def linear(x):
        return x


activations = {
    'sigmoid': (Activation.sigmoid, Activation.sigmoid_derivative),
    'relu': (Activation.relu, Activation.relu_derivative),
    'tanh': (Activation.tanh, Activation.tanh_derivative),
    'softmax': (Activation.softmax, None),  # Derivative handled in loss
    'linear': (Activation.linear, lambda x: np.ones_like(x))
}