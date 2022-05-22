import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


def create_new_dataset(dataset, seq_len=12):
    """ 基于原始数据集构造新的序列特征数据集
    Params:
        dataset : 原始数据集
        seq_len : 序列长度（时间跨度）
    """
    X = []  # 初始特征数据集为空列表
    y = []  # 初始标签数据集为空列表
    start = 0  # 初始位置
    end = dataset.shape[0] - seq_len  # 截止位置
    for i in range(start, end):  # for循环构造特征数据集
        sample = dataset[i: i + seq_len]  # 基于时间跨度seq_len创建样本
        label = dataset[i + seq_len]  # 创建sample对应的标签
        X.append(sample)  # 保存sample
        y.append(label)  # 保存label

    return np.array(X), np.array(y)


def split_dataset(X, y, test_len=None, train_ratio=0.8):
    """ 基于X和y，切分为train和test
    Params:
        X : 特征数据集
        y : 标签数据集
        train_ratio : 训练集占X的比例
    """
    X_len = len(X)  # 特征数据集X的样本数量
    if test_len is None:
        train_data_len = int(X_len * train_ratio)  # 训练集的样本数量
    else:
        train_data_len = X_len - test_len
    X_train = X[:train_data_len]  # 训练集
    y_train = y[:train_data_len]  # 训练标签集
    X_test = X[train_data_len:]  # 测试集
    y_test = y[train_data_len:]  # 测试集标签集

    return X_train, X_test, y_train, y_test


def create_batch_data(X, y, batch_size=32, data_type="train"):
    """ 基于训练集和测试集，创建批数据
     Params:
        X : 特征数据集
        y : 标签数据集
        batch_size : batch的大小，即一个数据块里面有几个样本
        data_type : 数据集类型（测试集表示"test"，训练集表示"train"）
    """
    if data_type == "test":  # 测试集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))  # 封装X和y，成为tensor类型
        test_batch_data = dataset.batch(batch_size)  # 构造批数据

        return test_batch_data
    else:  # 训练集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))  # 封装X和y，成为tensor类型
        train_batch_data = dataset.cache().shuffle(1000).batch(batch_size)  # 构造批数据

        return train_batch_data


def Sequential_Model(train_batch_dataset, test_batch_dataset):
    model = tf.keras.Sequential([
        LSTM(80, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='mean_squared_error',
                  metrics='accuracy')  # 损失函数用均方误差

    return model


def plot_loss_acc(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


# 预测结果可视化
def plot_predict(real, predict):
    plt.plot(real, color='red', label='illinois river no3')
    plt.plot(predict, color='blue', label='Predicted illinois river no3')
    plt.title('illinois river no3 Prediction')
    plt.xlabel('Time')
    plt.ylabel('illinois river no3')
    plt.legend()
    plt.show()


# 模型评估
def evaluate_model(real, predict):
    Max_Error = max_error(predict, real)
    MAE = mean_absolute_error(predict, real)
    MAPE = mean_absolute_percentage_error(predict, real)
    print('最大误差: %.6f' % Max_Error)
    print('平均绝对误差: %.6f' % MAE)
    print('平均绝对百分误差：%.6f' % MAPE)


if __name__ == '__main__':
    SEQ_LEN = 10  # 序列长度
    df = pd.read_csv('Illinois_3339000_2016-2021.csv')  # 读取文件
    scaler = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
    dataset = scaler.fit_transform(df['no3_mg_per_l'].values.reshape(-1, 1))
    # 创建数据集
    X, y = create_new_dataset(dataset, seq_len=SEQ_LEN)
    # 测试训练集分割
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_len=300)
    # 训练批数据和测试批数据
    test_batch_dataset = create_batch_data(X_test, y_test, batch_size=64, data_type="test")
    train_batch_dataset = create_batch_data(X_train, y_train, batch_size=64, data_type="train")
    # 加载序列模型
    model = Sequential_Model(train_batch_dataset, test_batch_dataset)
    # 模型的回调与保存
    checkpoint_save_path = "./checkpoint/LSTM_illinois_3346500_nO3.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     monitor='val_loss')
    # 模型训练
    history = model.fit(train_batch_dataset, batch_size=64, epochs=50,
                        validation_data=(test_batch_dataset),
                        validation_freq=1,
                        callbacks=[cp_callback])
    # 训练过程可视化
    plot_loss_acc(history)
    # 预测
    predict = scaler.inverse_transform(model.predict(X_test))
    y_test = np.reshape(y_test, (-1, 1))
    real = scaler.inverse_transform(y_test)
    # 预测结果可视化
    plot_predict(real, predict)
    # 模型评估
    evaluate_model(real, predict)
