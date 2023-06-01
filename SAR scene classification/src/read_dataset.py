
import numpy as np
import cv2
import os

import read_patch
def read_dataset_txt(txt_file):
    list_path_label = {}
    f_txt = open(txt_file)
    lines_txt = f_txt.readlines()

    for count, l in enumerate(lines_txt):
        l_split = l.split()
        l_path = l_split[0]
        l_label = l_split[1]

        list_path_label[count] = [l_path, int(l_label)]
    return list_path_label
def read_npy(patch_path):
    # patch = np.double(np.load(patch_path)) / 65535.0
    patch = np.load(patch_path)

    return patch

def label_smooth(cls_num, label_smooth, orign_label):
    label = np.zeros((10))
    lab = orign_label
    label[lab] = 1-label_smooth
    all_num = 0
    for i in range(len(label)):
        if label[i] == 0:
            all_num = all_num+cls_num[i]
    for i in range(len(label)):
        if label[i] == 0:
            label[i] = (cls_num[i]/all_num)*label_smooth
    return label