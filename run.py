# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/4/1 下午3:55
import glob
import os

from scipy import misc
import numpy as np

from networks import recognition_model


def main():
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
