
import numpy as np
import cv2
import os

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