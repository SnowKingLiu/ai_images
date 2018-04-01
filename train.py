# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/3/31 下午2:57
import glob
import os

from scipy import misc
from PIL import Image, ImageFile
import numpy as np
import tensorflow as tf

from networks import recognition_model


# 设置参数
ImageFile.LOAD_TRUNCATED_IMAGES = True
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA_1 = 0.5


def train():
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
    rec_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    # 配置识别网络
    rec.compile(loss="binary_crossentropy", optimizer=rec_optimizer)

    # 开始训练
    for epoch in range(10):
        for index in range(int(input_data.shape[0]/BATCH_SIZE)):
            input_batch = input_data[index*BATCH_SIZE: (index + 1)*BATCH_SIZE]
            output_batch = output_data[index*BATCH_SIZE: (index + 1)*BATCH_SIZE]
            rec_loss = rec.train_on_batch(input_batch, output_batch)

            # 打印损失
            print("第%d次 损失值是: %f" % (index, rec_loss))
        if epoch % 10 == 9:
            rec.save_weights("recognition_weight", True)


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
