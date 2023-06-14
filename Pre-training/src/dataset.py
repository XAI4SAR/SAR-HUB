import read_dataset
import os
import torch
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import random
import datetime
class TSX_Dataset(Dataset):
    """
    TerraSAR-X dataset
    """

    def __init__(self, txt_file, root_dir, data_type, transform = None):
        """
        Args:
            :param txt_file: path to the txt file with ($path$ $label$)
            :param root_dir: full_path = root_dir + file_path
            :param transform: optional transform to be applied on a sample
        """
        self.txt_file = txt_file
        self.tsx_path_label = read_dataset.read_dataset_txt(txt_file)
        self.root_dir = root_dir
        self.data_type = data_type
        self.transform = transform
    def __len__(self):
        return len(self.tsx_path_label)

    def __getitem__(self, idx):
        """
        Args:
            :param idx: the index of data
            :param data_type: "npy" or "tif"
            :return:
        """

        if self.data_type[0:3] == 'npy':
            patch_path = 'TSXdataset_npy_new/' + self.tsx_path_label[idx][0] + '.npy'
            image = read_dataset.read_npy(patch_path)
        elif self.data_type == 'tif':
            patch_path = os.path.join(self.tsx_path_label[idx][0]) + '.jpg'
            image = read_dataset.read_tif(self.root_dir + '/TSXdatasetOctavian', patch_path)
        elif self.data_type == 'jpg':
            patch_path = os.path.join(self.root_dir, 'TSXdatasetOctavian', self.tsx_path_label[idx][0]) + '.jpg'
            image = read_dataset.read_jpg(patch_path)
        # print(self.tsx_path_label)
        label = self.tsx_path_label[idx][1]
        # print(label)
        sample = {'image' : image, 'label' : label, 'path': self.tsx_path_label[idx][0],'index':idx}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

class BEN_Dataset(Dataset):
    """
    BigEarthNet-S1 dataset
    """

    def __init__(self, txt_file, root_dir, data_type, transform = None):
        """
        Args:
            :param txt_file: path to the txt file with ($path$ $label$)
            :param root_dir: full_path = root_dir + file_path
            :param transform: optional transform to be applied on a sample
        """
        self.txt_file = txt_file
        self.tsx_path_label = read_dataset.read_dataset_txt(txt_file)
        self.root_dir = root_dir
        self.data_type = data_type
        self.transform = transform
    def __len__(self):
        return len(self.tsx_path_label)

    def __getitem__(self, idx):
        """
        Args:
            :param idx: the index of data
            :param data_type: "npy" or "tif"
            :return:
        """

        patch_path = 'BigEarthNet/BEN-S1-npy/' + self.tsx_path_label[idx][0]
        image = read_dataset.read_npy(patch_path)
        # print(self.tsx_path_label)
        lab = np.zeros((19),dtype = 'float32')    #! one hot
        label = self.tsx_path_label[idx][1]
        for i in label:
            lab[int(i)] = 1
        # print(label)
        sample = {'image' : image, 'label' : lab, 'path': self.tsx_path_label[idx][0],'index':idx}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample
    
class OSU_Dataset(Dataset):
    """
    OpenSARurban dataset
    """

    def __init__(self, txt_file, root_dir, data_type, transform = None):
        """
        Args:
            :param txt_file: path to the txt file with ($path$ $label$)
            :param root_dir: full_path = root_dir + file_path
            :param transform: optional transform to be applied on a sample
        """
        self.txt_file = txt_file
        self.osu_path_label = read_dataset.read_dataset_txt(txt_file)
        self.root_dir = root_dir
        self.data_type = data_type
        self.transform = transform
        # self.cls_num = cls_num
    def __len__(self):
        return len(self.osu_path_label)

    def __getitem__(self, idx):
        """
        Args:
            :param idx: the index of data
            :param data_type: "npy" or "tif"
            :return:
        """

        patch_path = 'OpenSARurban/OSU-npy/' + self.osu_path_label[idx][0]
        image = read_dataset.read_npy(patch_path)
        label = self.osu_path_label[idx][1]
        

        sample = {'image' : image, 'label' : label, 'path': self.osu_path_label[idx][0],'index':idx}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

