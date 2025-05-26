# MLP 实现

### 描述
本项目主要是对MLP的实现。不依赖现有的神经网络框架，纯python实现

### 项目结构

├── activations.py       # 激活函数   
├── confusion_matrix.png # 混淆矩阵  
├── layers.py            # 网络层   
├── log.txt              # 训练日志  
├── losses.py            # 损失函数  
├── mnist_model.pkl      # 训练模型  
├── mnist_test.csv       # 测试数据集 文件过大未上传
├── mnist_train.csv      # 训练数据集  文件过大未上传
├── model.py             # MLP模型  
├── optimizers.py        # 优化器  
├── regularizers.py      # 正则化  
├── train.py             # 训练类  
└── utils.py             # 工具类  

### 训练
直接运行train.py


### 评估
train.py中包含训练之后的评估，以生成混淆矩阵的方式