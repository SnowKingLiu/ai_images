# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/3/31 下午2:57
import glob
import os

from matplotlib import pyplot
from scipy import misc
from PIL import Image, ImageFile
import numpy as np
import tensorflow as tf

from networks import recognition_model, cifar_model

# 设置参数
from utils import unpickle, list2onehot

ImageFile.LOAD_TRUNCATED_IMAGES = True
EPOCHS = 1000000
BATCH_SIZE = 2048
LEARNING_RATE = 0.0002
BETA_1 = 0.5


def train():
    dic1 = unpickle("data_batch_1")
    dic2 = unpickle("data_batch_2")
    dic3 = unpickle("data_batch_3")
    dic4 = unpickle("data_batch_4")
    dic5 = unpickle("data_batch_5")
    # data1 = dic1[b'data']
    input_data = np.array([arr.reshape(32, 32, 3) for arr in list(dic1[b'data']) + list(dic2[b'data']) +
                           list(dic3[b'data']) + list(dic4[b'data']) + list(dic5[b'data'])])
    output_data = list2onehot(np.array(dic1[b'labels'] + dic2[b'labels'] + dic3[b'labels'] + dic4[b'labels'] +
                                       dic5[b'labels']))
    cif = cifar_model()
    if os.path.isfile("cifar_weight.hdf5"):
        cif.load_weights("cifar_weight.hdf5")
    rec_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    # 配置识别网络
    cif.compile(loss="categorical_crossentropy", optimizer=rec_optimizer)

    # 开始训练
    check_pointer = tf.keras.callbacks.ModelCheckpoint(filepath="cifar_weight.hdf5", verbose=1,
                                                       save_best_only=True)
    cif.fit(x=input_data, y=output_data, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle="batch",
            validation_data=(input_data, output_data), callbacks=[check_pointer])


def get_img(arr):
    # 得到RGB通道
    r = Image.fromarray(arr[0]).convert('L')
    g = Image.fromarray(arr[1]).convert('L')
    b = Image.fromarray(arr[2]).convert('L')
    image = Image.merge("RGB", (r, g, b))
    # 显示图片
    pyplot.imshow(image)
    pyplot.show()


def train_clothing():
    # 获取训练数据
    image_list = []
    label_list = []
    for image in glob.glob("newImg/男装/*.jpg"):
        # imread利用PIL来读取图片数据
        image_data = misc.imread(image)
        image_list.append(image_data)
        label_list.append(np.array([1, 0]))
    for image in glob.glob("newImg/女装/*.jpg"):
        # imread利用PIL来读取图片数据
        image_data = misc.imread(image)
        image_list.append(image_data)
        label_list.append(np.array([0, 1]))
    input_data = np.array(image_list)
    output_data = np.array(label_list)

    rec = recognition_model()

    if os.path.isfile("recognition_weight.hdf5"):
        rec.load_weights("recognition_weight.hdf5")
    rec_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    # 配置识别网络
    rec.compile(loss="categorical_crossentropy", optimizer=rec_optimizer)

    # 开始训练
    check_pointer = tf.keras.callbacks.ModelCheckpoint(filepath="recognition_weight.hdf5", verbose=1,
                                                       save_best_only=True)
    rec.fit(x=input_data, y=output_data, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle="batch",
            validation_data=(input_data, output_data), callbacks=[check_pointer])

    rec.predict(input_data[0:2])
    # for index in range(int(input_data.shape[0]/BATCH_SIZE)):
    #     input_batch = input_data[index*BATCH_SIZE: (index + 1)*BATCH_SIZE]
    #     output_batch = output_data[index*BATCH_SIZE: (index + 1)*BATCH_SIZE]
    #     rec.fit(x=input_batch, y=output_batch, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle="batch")
    #     # rec_loss = rec.train_on_batch(input_batch, output_batch)
    # 打印损失
    # print("第%d次 损失值是: %f" % (epoch, rec_loss))
    # if epoch % 10 == 9:
    #     rec.save_weights("recognition_weight", True)


def get_img():
    # 获取训练数据
    size = (64, 64)
    i = 0
    for dir1 in os.listdir(os.path.join(os.path.dirname(__file__), "images/女装")):
        for image in glob.glob("images/女装/{}/*".format(dir1)):
            img = Image.open(image).convert('RGB')
            img.resize(size).save("newImg/女装/{}{}".format(str(i).zfill(4), ".jpg"))
            i += 1


if __name__ == '__main__':
    train()
