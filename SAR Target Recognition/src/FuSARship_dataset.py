from torch.utils.data import DataLoader,Subset,Dataset
import os
import torch
import numpy as np
from PIL import Image

class FuSARshipDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        with open(txt, 'r') as fh:
            imgs = []
            for line in fh:
                line = line.strip('\n')  
                line = line.rstrip()  
                words = line.split()  
                imgs.append((words[0], int(words[1])))  
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