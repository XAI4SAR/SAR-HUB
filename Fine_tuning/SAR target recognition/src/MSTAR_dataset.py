import read_dataset
import os
import torch
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import random
import datetime

    
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
        patch_path = 'transfer/SAR_Data/MSTAR_chip/SOC' + self.path_label[idx][0]
        image = read_dataset.read_npy(patch_path)
        label = self.path_label[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, label