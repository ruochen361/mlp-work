import numpy as np
from losses import loss_functions
from mlp.layers import DenseLayer
from regularizers import Regularizer


class MLPModel:
    def __init__(self, layers, activations, task='classification',
                 weight_init='xavier', reg=None, reg_lambda=0.0):
        self.layers = []
        self.task = task
        self.reg = reg
        self.reg_lambda = reg_lambda
        self.loss_history = []

        # Build layers
        for i in range(len(layers) - 1):
            layer = DenseLayer(
                input_size=layers[i],
                output_size=layers[i + 1],
                activation=activations[i],
                weight_init=weight_init,
                reg=reg,
                reg_lambda=reg_lambda
            )
            self.layers.append(layer)

        # Set output activation based on task
        if task == 'classification':
            self.output_activation = 'softmax'
            self.loss_fn, self.loss_deriv = loss_functions['cross_entropy']
        else:
            self.output_activation = 'linear'
            self.loss_fn, self.loss_deriv = loss_functions['mse']

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compute_loss(self, y_true, y_pred):
        loss = self.loss_fn(y_true, y_pred)
        # Add regularization loss
        if self.reg:
            reg_loss = 0
            for layer in self.layers:
                if self.reg == 'l1':
                    reg_loss += Regularizer.l1(layer.weights, self.reg_lambda)
                elif self.reg == 'l2':
                    reg_loss += Regularizer.l2(layer.weights, self.reg_lambda)
                elif self.reg == 'elastic':
                    reg_loss += Regularizer.elastic(layer.weights, self.reg_lambda)
            loss += reg_loss
        return loss

    def backward(self, y_true, y_pred, optimizer):
        dA = self.loss_deriv(y_true, y_pred)
        for layer in reversed(self.layers):
            dA = layer.backward(dA, optimizer)

    def train(self, X, y, epochs=100, batch_size=32, optimizer=SGD(),
              validation_data=None, stop_criteria=None, verbose=True):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += loss

                # Backward pass
                self.backward(y_batch, y_pred, optimizer)

            # Track loss
            avg_loss = epoch_loss / (n_samples / batch_size)
            self.loss_history.append(avg_loss)

            # Check stopping criteria
            if stop_criteria and stop_criteria(self):
                break

            if verbose:
                print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

    def predict(self, X):
        y_pred = self.forward(X)
        if self.task == 'classification':
            return np.argmax(y_pred, axis=1)
        return y_pred