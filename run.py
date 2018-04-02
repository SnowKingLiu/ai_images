# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/4/1 下午3:55
import glob
import os

from scipy import misc
import numpy as np

from networks import recognition_model, cifar_model
from utils import unpickle, list2onehot


def main():
    cif = cifar_model()
    if os.path.isfile("cifar_weight.hdf5"):
        cif.load_weights("cifar_weight.hdf5")
    else:
        print("获取参数失败！")
        return
    dic = unpickle("test_batch")
    data = dic[b'data']
    input_data = np.array([arr.reshape(32, 32, 3) for arr in data])
    output_data = list2onehot(np.array(dic[b'labels']))
    # 编译模型
    cif.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    scores = cif.evaluate(input_data, output_data, verbose=0)
    print("损失值为：%.2f%%" % (scores[1]*100))


def predict_clothing():
    rec = recognition_model()
    if os.path.isfile("recognition_weight.hdf5"):
        rec.load_weights("recognition_weight.hdf5")
    else:
        print("获取参数失败！")
        return
    # 加载训练集
    image_list = []
    label_list = []
    for image in glob.glob("test/男装/*.jpg"):
        # imread利用PIL来读取图片数据
        image_data = misc.imread(image)
        image_list.append(image_data)
        label_list.append(np.array([1, 0]))
    input_data = np.array(image_list)
    output_data = np.array(label_list)
    print(rec.predict(input_data))
    print(output_data)
    pass


if __name__ == '__main__':
    main()
