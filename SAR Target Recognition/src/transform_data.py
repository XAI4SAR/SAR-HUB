
import numpy as np
import cv2
import random
import torch
from torchvision import transforms
from transform import MaxMin, Reinhard_Devlin,zeroTo65535,Linear,gamma,trans
"""
These callable classes are all for SAR images which have only 1 channel and read as np.array [0, 1]
含有MAX-MIN变换方式以及0-65535变换方式
Log变换，linear线性变换和gamma线性变换
"""


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (img - self.mean) / self.std

        return img
class Normalize1(object):
    def __call__(self, img):
        mean = np.mean(img)
        var = np.var(img)
        return ((img-mean)/var)
class zero8(object):
    def __call__(self, img):
        return img/255

class Rescale(object):

    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size

    def __call__(self, img):
      th, tw = self.size
      return cv2.resize(img, (th, tw))


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        img_h, img_w = img.shape
        start_x = random.randint(0,img_w-tw)
        start_y = random.randint(0,img_h-th)
        return img[start_y:start_y+th, start_x:start_x+tw]

class PlaceCrop(object):
    """Crops the given np.array at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (np.array): Image to be cropped.
        Returns:
            np.array: Cropped image.
        """
        th, tw = self.size
        return img[self.start_y : self.start_y + th, self.start_x : self.start_x + tw]

class RandomFlip(object):
    """Horizontally flip the given np.array randomly with a probability of 0.5 """

    def __call__(self, img):
        flag = random.randint(0,1)
        if flag:
            return img[:, ::-1]
        else:
            return img


class LogTransform(object):
    def __call__(self, img):
        return np.log2(np.double(img + 1)) / 16

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
            return img_copy.float()

def image_test_5crop(nor_mean, nor_std, resize_size, crop_size):

  #ten crops for image when validation, input the data_transforms dictionary
  start_first = 0
  start_center = int((resize_size - crop_size - 1) / 2)
  start_last = resize_size - crop_size - 1
  data_transforms = {}
  data_transforms['val0'] = transforms.Compose([
      LogTransform(),
      Normalize(nor_mean, nor_std),
      Rescale(resize_size),
      PlaceCrop(crop_size, start_first, start_first),
      Numpy2Tensor(3)
  ])
  data_transforms['val1'] = transforms.Compose([
      LogTransform(),
      Normalize(nor_mean, nor_std),
      Rescale(resize_size),
      PlaceCrop(crop_size, start_last, start_last),
      Numpy2Tensor(3)
  ])
  data_transforms['val2'] = transforms.Compose([
      LogTransform(),
      Normalize(nor_mean, nor_std),
      Rescale(resize_size),
      PlaceCrop(crop_size, start_last, start_first),
      Numpy2Tensor(3)
  ])
  data_transforms['val3'] = transforms.Compose([
      LogTransform(),
      Normalize(nor_mean, nor_std),
      Rescale(resize_size),
      PlaceCrop(crop_size, start_first, start_last),
      Numpy2Tensor(3)
  ])
  data_transforms['val4'] = transforms.Compose([
      LogTransform(),
      Normalize(nor_mean, nor_std),
      Rescale(resize_size),
      PlaceCrop(crop_size, start_center, start_center),
      Numpy2Tensor(3)
  ])

  return data_transforms

class MAXMIN(object):
    def __call__(self, img):
        return MaxMin(img)

class zero16(object):
    def __call__(self, img):
        return zeroTo65535(img)

class Linr(object):
    def __call__(self, img):
        return Linear(img)

class Gamma(object):
    def __call__(self, img):
        return gamma(img)

class Drag(object):
    def __init__(self,c,b,m):
        self.c = c
        self.b = b
        self.m = m
    def __call__(self, img):
        return trans(img,self.c,self.b,self.m)
class Reinhard(object):
    def __call__(self, img):
        return Reinhard_Devlin(img)