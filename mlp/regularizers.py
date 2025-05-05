import numpy as np

# 正则化

class Regularizer:
    @staticmethod
    def l1(weights, lambda_):
        return lambda_ * np.sum(np.abs(weights))

    @staticmethod
    def l1_grad(weights, lambda_):
        return lambda_ * np.sign(weights)

    @staticmethod
    def l2(weights, lambda_):
        return 0.5 * lambda_ * np.sum(weights ** 2)

    @staticmethod
    def l2_grad(weights, lambda_):
        return lambda_ * weights

    @staticmethod
    def elastic(weights, lambda_, l1_ratio=0.5):
        return l1_ratio * Regularizer.l1(weights, lambda_) + (1 - l1_ratio) * Regularizer.l2(weights, lambda_)

    @staticmethod
    def elastic_grad(weights, lambda_, l1_ratio=0.5):
        return l1_ratio * Regularizer.l1_grad(weights, lambda_) + (1 - l1_ratio) * Regularizer.l2_grad(weights, lambda_)