'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-03-22 15:06:34
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-04-17 20:59:24
FilePath: /pycharm_YHD/knowledge-point/src/MSTARdataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import read_dataset
import os
import torch
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import random
import datetime


class Mstar(Dataset):
    def __init__(self, list_dir, transform): #self是全局变量,其他地方书写时不需要加self
        self.data_list = {'npz_list':[], 'label_list':[]} 
        self.transform = transform
        f = open(list_dir, 'r')
        for i in f.readlines():
            self.data_list['npz_list'].append(i.strip().split()[0])
            self.data_list['label_list'].append(int(i.strip().split()[1]))

    def __getitem__(self, idx):
        npz_path = self.data_list['npz_list'][idx]
        mag_img = abs(np.load(npz_path)['comp'])
        if self.transform:
            mag_img = self.transform(mag_img)
        return mag_img, self.data_list['label_list'][idx]

    def __len__(self):
        return len(self.data_list['npz_list']) #返回列表长度
    
class MsTAR_Dataset(Dataset):
    '''
    MSTAR
    '''
    def __init__(self, txt_file, transform = None):
        self.txt_file = txt_file
        self.path_label = read_dataset.read_dataset_txt(txt_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.path_label)
    
    def __getitem__(self, idx):
        patch_path = '/knowledge-point/SOC' + self.path_label[idx][0]
        # patch_path = '/home/hzl/STAT2/YHD/pycharm_YHD/knowledge-point/SOC' + self.path_label[idx][0]
        image = read_dataset.read_npy(patch_path)
        label = self.path_label[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label