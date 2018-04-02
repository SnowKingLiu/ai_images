# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/3/31 下午2:19
import tensorflow as tf


def recognition_model():
    rec_model = tf.keras.models.Sequential()

    # 卷积层
    rec_model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', input_shape=(64, 64, 3)))
    # 添加激活函数
    rec_model.add(tf.keras.layers.Activation("relu"))
    # 池化层
    rec_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # 卷积层
    rec_model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    # 添加激活函数
    rec_model.add(tf.keras.layers.Activation("relu"))
    # 池化层
    rec_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # 卷积层
    rec_model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    # 添加激活函数
    rec_model.add(tf.keras.layers.Activation("relu"))
    # 池化层
    rec_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # 扁平化
    rec_model.add(tf.keras.layers.Flatten())

    # 1024个神经元的全连接层
    rec_model.add(tf.keras.layers.Dense(1024))
    # 丢弃其中的50%
    rec_model.add(tf.keras.layers.Dropout(0.5))
    # 添加激活函数
    rec_model.add(tf.keras.layers.Activation("relu"))
    # 2个神经元的全连接层
    rec_model.add(tf.keras.layers.Dense(2))
    # 添加Sigmoid激活层
    rec_model.add(tf.keras.layers.Activation("softmax"))

    return rec_model


def cifar_model():
    cif_model = tf.keras.Sequential()

    # 卷积层
    cif_model.add(tf.keras.layers.Conv2D(32, (5, 5), padding='same', input_shape=(32, 32, 3)))
    # 添加激活函数
    cif_model.add(tf.keras.layers.Activation("relu"))
    # 池化层
    cif_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # 卷积层
    cif_model.add(tf.keras.layers.Conv2D(64, (5, 5)))
    # 添加激活函数
    cif_model.add(tf.keras.layers.Activation("relu"))
    # 池化层
    cif_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # 卷积层
    cif_model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    # 添加激活函数
    cif_model.add(tf.keras.layers.Activation("relu"))
    # 池化层
    cif_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    # 扁平化
    cif_model.add(tf.keras.layers.Flatten())

    # 1024个神经元的全连接层
    cif_model.add(tf.keras.layers.Dense(1024))
    # 丢弃其中的50%
    cif_model.add(tf.keras.layers.Dropout(0.5))
    # 添加激活函数
    cif_model.add(tf.keras.layers.Activation("relu"))
    # 2个神经元的全连接层
    cif_model.add(tf.keras.layers.Dense(10))
    # 添加Sigmoid激活层
    cif_model.add(tf.keras.layers.Activation("softmax"))

    return cif_model


if __name__ == '__main__':
    model = recognition_model()
    model.summary()
