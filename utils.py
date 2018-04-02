# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2018/4/2 下午4:51
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def list2onehot(int_list):
    one_hot = np.zeros((len(int_list), len(set(int_list))))
    for i in range(len(int_list)):
        one_hot[i][int_list[i]] = 1
    return one_hot
