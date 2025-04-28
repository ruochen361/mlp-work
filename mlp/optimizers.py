import numpy as np

# 优化器

class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, layer):
        layer.weights -= self.lr * layer.dW
        layer.biases -= self.lr * layer.db


class Momentum(SGD):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum

    def update(self, layer):
        if layer.velocity_w is None:
            layer.velocity_w = np.zeros_like(layer.weights)
            layer.velocity_b = np.zeros_like(layer.biases)

        layer.velocity_w = self.momentum * layer.velocity_w - self.lr * layer.dW
        layer.velocity_b = self.momentum * layer.velocity_b - self.lr * layer.db
        layer.weights += layer.velocity_w
        layer.biases += layer.velocity_b


class RMSProp(SGD):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon

    def update(self, layer):
        if layer.v_w is None:
            layer.v_w = np.zeros_like(layer.weights)
            layer.v_b = np.zeros_like(layer.biases)

        layer.v_w = self.beta * layer.v_w + (1 - self.beta) * layer.dW ** 2
        layer.v_b = self.beta * layer.v_b + (1 - self.beta) * layer.db ** 2
        layer.weights -= self.lr * layer.dW / (np.sqrt(layer.v_w) + self.epsilon)
        layer.biases -= self.lr * layer.db / (np.sqrt(layer.v_b) + self.epsilon)


class Adam(SGD):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def update(self, layer):
        self.t += 1
        if layer.m_w is None:
            layer.m_w = np.zeros_like(layer.weights)
            layer.v_w = np.zeros_like(layer.weights)
            layer.m_b = np.zeros_like(layer.biases)
            layer.v_b = np.zeros_like(layer.biases)

        layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * layer.dW
        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.db
        layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * layer.dW ** 2
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * layer.db ** 2

        m_w_hat = layer.m_w / (1 - self.beta1 ** self.t)
        m_b_hat = layer.m_b / (1 - self.beta1 ** self.t)
        v_w_hat = layer.v_w / (1 - self.beta2 ** self.t)
        v_b_hat = layer.v_b / (1 - self.beta2 ** self.t)

        layer.weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer.biases -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)