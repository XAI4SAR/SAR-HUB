import numpy as np
import cv2
import random
import torch
from torchvision import transforms
from transform import MaxMin, Reinhard_Devlin_train, Reinhard_Devlin_test,zeroTo65535,Linear,gamma,trans,truncate_train_osu,truncate_train_ben,truncate_test_osu,truncate_test_ben
import math


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (img - self.mean) / self.std

        return img

class Numpy2Tensor(object):
    """Convert a 1-channel ``numpy.ndarray`` to 1-c or 3-c tensor,
    depending on the arg parameter of "channels"
    """
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, img):
        """
        for SAR images (.npy), shape H * W, we should transform into C * H * W
        :param img:
        :return:
        """
        channels = self.channels
        img_copy = np.zeros([channels, img.shape[0], img.shape[1]])

        for i in range(channels):
            img_copy[i, :, :] = np.reshape(img, [1, img.shape[0], img.shape[1]]).copy()

        if not isinstance(img_copy, np.ndarray) and (img_copy.ndim in {2, 3}):
            raise TypeError('img should be ndarray. Got {}'.format(type(img_copy)))

        if isinstance(img_copy, np.ndarray):
            # handle numpy array
            img_copy = torch.from_numpy(img_copy)
            # backward compatibility
            # img_copy = img_copy.type(torch.ByteTensor)
            # print(img_copy)
            return img_copy.float()
        
class Reinhard_train(object):
    def __call__(self, img):
        return Reinhard_Devlin_train(img)
class Reinhard_test(object):
    def __call__(self, img):
        return Reinhard_Devlin_test(img)
    
class truncated_train_osu(object):
    def __call__(self, img):
        return truncate_train_osu(img,255,0)
class truncated_test_osu(object):
    def __call__(self, img):
        return truncate_test_osu(img,255,0)
    
class truncated_train_ben(object):
    def __call__(self, img):
        return truncate_train_ben(img,255,0)
class truncated_test_ben(object):
    def __call__(self, img):
        return truncate_test_ben(img,255,0)
