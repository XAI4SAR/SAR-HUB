import read_dataset
import os
import torch
import time
import numpy as np
import random
import datetime
from torch.utils.data import DataLoader,Subset,Dataset
from PIL import Image

class FuSARshipDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        with open(txt, 'r') as fh:
            imgs = []
            for line in fh:
                line = line.strip('\n')  # 移除字符串首尾的换行符
                line = line.rstrip()  # 删除末尾空
                words = line.split()  # 以空格为分隔符 将字符串分成
                imgs.append((words[0], int(words[1])))  # imgs中包含有图像路径和标签
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = np.load(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class OpenSARshipDataset(Dataset):
    def __init__(self,txt, transform=None,size='OpenSARship_low'):
        with open(txt, 'r') as fh:
            imgs = []
            for line in fh:
                line = line.strip('\n')  # 移除字符串首尾的换行符
                line = line.rstrip()  # 删除末尾空
                words = line.split()  # 以空格为分隔符 将字符串分成
                imgs.append((words[0], int(words[1])))  # imgs中包含有图像路径和标签
        self.imgs = imgs
        self.transform = transform
        self.size = size

    def __getitem__(self,index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('L')
        if self.size == 'OpenSARship_low':
            img = img.resize((32,32))
        elif self.size == 'OpenSARship_medium':
            img = img.resize((96,96))
        elif self.size == 'OpenSARship_large':
            img = img.resize((128,128))
        img=np.copy(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
    
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
        # patch_path = '/DATA/yhd/pycharm_YHD/transfer/target_MSTAR/MSTAR_chip/SOC' + self.path_label[idx][0]
        patch_path = '/DATA/yhd/pycharm_YHD/transfer/SAR_Data/MSTAR_chip/SOC' + self.path_label[idx][0]
        image = read_dataset.read_npy(patch_path)
        label = self.path_label[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label
    
