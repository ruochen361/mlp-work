import numpy as np

from mlp.activations import activations
from mlp.regularizers import Regularizer


class DenseLayer:
    def __init__(self, input_size, output_size, activation, weight_init, reg=None, reg_lambda=0.0):
        self.weights = self.initialize_weights(input_size, output_size, weight_init)
        self.biases = np.zeros((1, output_size))
        self.activation_name = activation
        self.activation, self.activation_deriv = activations[activation]
        self.reg = reg
        self.reg_lambda = reg_lambda

        # Optimizer states
        self.m_w = self.v_w = self.m_b = self.v_b = None  # For Adam/RMSProp
        self.velocity_w = self.velocity_b = None  # For Momentum

    def initialize_weights(self, input_size, output_size, method):
        if method == 'xavier':
            # Xavier初始化  适用Sigmoid、Tanh 等对称且梯度在原点附近较大的激活函数
            limit = np.sqrt(6 / (input_size + output_size))
            return np.random.uniform(-limit, limit, (input_size, output_size))
        elif method == 'he':
            # He初始化 适用ReLU、Leaky ReLU、PReLU 等非对称或具有“死区”的激活函数
            std = np.sqrt(2 / input_size)
            return np.random.randn(input_size, output_size) * std
        else:  # 'normal'
            return 0.01 * np.random.randn(input_size, output_size)

    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.weights) + self.biases
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, dA, optimizer):
        if self.activation_deriv:
            dZ = dA * self.activation_deriv(self.Z)
        else:
            dZ = dA  # For softmax handled in loss

        m = self.X.shape[0]
        self.dW = np.dot(self.X.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m

        # Add regularization gradient
        if self.reg == 'l1':
            self.dW += Regularizer.l1_grad(self.weights, self.reg_lambda)
        elif self.reg == 'l2':
            self.dW += Regularizer.l2_grad(self.weights, self.reg_lambda)
        elif self.reg == 'elastic':
            self.dW += Regularizer.elastic_grad(self.weights, self.reg_lambda)

        optimizer.update(self)

        return np.dot(dZ, self.weights.T)