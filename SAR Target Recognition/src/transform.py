'''
Author: your name
Date: 2022-04-20 11:12:42
LastEditTime: 2022-04-20 11:12:42
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /pycharm_YHD/transfer/target_MSTAR/src/transform.py
'''
'''
Author: your name
Date: 2021-07-06 20:42:24
LastEditTime: 2021-12-24 11:35:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \teammate\TSX_EXP\TSX\TSX_code\src\transform.py
'''
import torch
import numpy as np
from torchvision import transforms
import math
from math import log
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random
def MaxMin(image_np):
    '''
    :param image_np: 对图形numpy进行最大最小归一
    :return: 返回处理后的图像numpy
    '''
    return (image_np-np.min(image_np))/(np.max(image_np)-np.min(image_np))

def zeroTo65535(image_np):
    '''
    :param image_np: 需要处理的归一化图像的numpy，0-65535/2归一
    :return: 返回处理后的图像numpy
    '''
    return (image_np/(65535/2))

def Linear(image_np):
    '''
    :param image_np: 对图像数组进行线性处理，此处应处理归一化后的图像
    :return: 返回处理后的图像数组，用Ldimg来表示
    '''
    # print(image_np)
    Img_mean = np.mean(image_np)
    Img_std = np.std(image_np)
    t = Img_mean+3*Img_std
    # Ldimg = np.zeros(np.shape(image_np))
    # for index,value in np.ndenumerate(image_np):
    #     if image_np[index] > t:
    #         Ldimg[index] = t
    #     else:
    #         Ldimg[index] = image_np[index]
    Ldimg = np.where(image_np>t,t,image_np)
    Ldimg = Ldimg/t
    return Ldimg

def gamma(image_np):
    '''
    :param image_np: 对图像数组进行gamma变化处理，此处应处理归一化后的图像
    :return: 返回处理后的图像数组，用gamimg表示
    '''
    # gamma = input('请输入伽马值：')
    # gamma = float(gamma)
    
    gamma = 0.90
    # for i in range(data_cols):
    
    # t = np.where((gamimg >= 0) & (gamimg <= 2.5*m), (gamimg/(65535/2)), 255)
    return image_np**gamma

def DragoLogMapping(img, c, b, m):
    '''
    :param img: 需要处理的sar图像数组
    :param c: Log函数的参数
    :param b: 控制图像对比度
    :param m: 控制图像整体亮度
    :return: 返回经过处理后的图像数组
    '''
    X = m/ log(1+c)
    img = X * (np.log(1+c*img))/(np.log(2+8*np.power(img, b)))
    return img

def trans(img,c,b,m):
    '''
    利用Drag对数公式、灰度直方均衡化与限制对比度的自适应直方图均衡化对图像进行变换
    :param img:需要变换的图像数组
    :return:返回变换之后的图像数组
    '''
    X = DragoLogMapping(img,c,b,m) #5 0.2 100; 150 -0.3 100; 差距不能太大
    X = Image.fromarray(X)
    X = X.convert('L')
    img1 = np.array(X)
    equ = cv2.equalizeHist(img1)
    # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    # cl1 = clahe.apply(equ)
    return equ