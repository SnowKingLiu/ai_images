# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/3/31 下午2:19
import tensorflow as tf


def recognition_model():
    model = tf.keras.models.Sequential()

    # 卷积层
    model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', input_shape=(64, 64, 3)))
    # 添加激活函数
    model.add(tf.keras.layers.Activation("relu"))
    # 池化层
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # 卷积层
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    # 添加激活函数
    model.add(tf.keras.layers.Activation("relu"))
    # 池化层
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # 卷积层
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    # 添加激活函数
    model.add(tf.keras.layers.Activation("relu"))
    # 池化层
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # 扁平化
    model.add(tf.keras.layers.Flatten())

    # 1024个神经元的全连接层
    model.add(tf.keras.layers.Dense(1024))
    # 丢弃其中的50%
    model.add(tf.keras.layers.Dropout(0.5))
    # 添加激活函数
    model.add(tf.keras.layers.Activation("relu"))
    # 2个神经元的全连接层
    model.add(tf.keras.layers.Dense(2))
    # 添加Sigmoid激活层
    model.add(tf.keras.layers.Activation("sigmoid"))

    return model
