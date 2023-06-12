from torch.utils.data import DataLoader,Subset,Dataset
import os
import numpy as np
from PIL import Image

class OpenSARshipDataset(Dataset):
    def __init__(self, txt, transform=None):
        with open(txt, 'r') as fh:
            imgs = []
            for line in fh:
                line = line.strip('\n') 
                line = line.rstrip()  
                words = line.split()  
                imgs.append((words[0], int(words[1])))  
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('L')
        img = img.resize((128,128)) # large
        # medium img = img.resize((96,96))  
        # low img = img.resize((32,32))
        img=np.copy(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)