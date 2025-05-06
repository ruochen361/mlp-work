import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from model import MLPModel
from optimizers import Adam
from utils import load_mnist, plot_confusion_matrix

# mnist数据集
X_train, y_train, X_test, y_test = load_mnist(
    "mnist_train.csv",
    "mnist_test.csv"
)

#  检查数据
print("输入数据维度:", X_train.shape)  # 应为 (60000, 784)
print("标签维度:", y_train.shape)     # 应为 (60000, 10)
print("样本0的标签:", y_train[0].argmax())  # 应输出合理数字

# Initialize model
model = MLPModel(
    layers=[784, 128, 64, 10],
    activations=['relu', 'relu', 'softmax'],
    task='classification',
    weight_init='he',
    reg=None,
    reg_lambda=0.01
)

# Train
model.train(
    X_train,
    y_train,
    epochs=10,
    batch_size=128,
    optimizer=Adam(learning_rate=0.0001),
    validation_data=(X_test, y_test),
    verbose=True
)

# 保存模型到文件
joblib.dump(model, 'mnist_model.pkl')

# Evaluate
test_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, test_pred)
# conf_matrix = confusion_matrix(y_test, test_pred)
plot_confusion_matrix(conf_matrix, 10)
print(f"Test Accuracy: {np.mean(test_pred == y_test):.4f}")