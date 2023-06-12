'''
Author: your name
Date: 2021-07-06 20:42:24
LastEditTime: 2023-04-20 10:35:55
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
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

def gamma(image_np,gamma):
    '''
    :param image_np: 对图像数组进行gamma变化处理，此处应处理归一化后的图像
    :return: 返回处理后的图像数组，用gamimg表示
    '''
    # gamma = input('请输入伽马值：')
    # gamma = float(gamma)
    
    # gamma = 0.90
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

def Reinhard_Devlin_train(img):
    amp_img = img
    # amp_img = (amp_img-amp_img.min()) / (amp_img.max()-amp_img.min())
    u = 0.168605
    # u = 0.168605
    # u = np.mean(img)
    l = 0.0001              # l范围:[0,1]，l越大，图像整体加灰 l最适：0.001,0.0001,0.00001
    b = random.uniform(3.5, 4.5)
    img_a = l * amp_img + (1 - l) * u
    m = 0.3 + 0.7 * np.power(((1 - u) / (1 - np.min(amp_img))), 1.4)
    img_d = (amp_img / (amp_img + np.power((b * img_a), m)))
    return img_d
def Reinhard_Devlin_test(img):
    amp_img = img
    # amp_img = (amp_img-amp_img.min()) / (amp_img.max()-amp_img.min())
    # u = random.uniform(0.168718,0.168723)
    u = 0.169
    # u = np.mean(img)
    l = 0.0001              # l范围:[0,1]，l越大，图像整体加灰 l最适：0.001,0.0001,0.00001
    b = 4
    img_a = l * amp_img + (1 - l) * u
    m = 0.3 + 0.7 * np.power(((1 - u) / (1 - np.min(amp_img))), 1.4)
    img_d = (amp_img / (amp_img + np.power((b * img_a), m)))
    return img_d
def truncate_train_osu(image, max_out = 255, min_out = 0):
    def gray_process(gray):
        truncated_value = random.uniform(0,2)
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out 
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        if(max_out <= 255):
            gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        elif(max_out <= 65535):
            gray = cv2.normalize(gray, gray, 0, 65535, cv2.NORM_MINMAX)
        return (gray-gray.min())/(gray.max()-gray.min())
    
    #  如果是多波段
    if(len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    #  如果是单波段
    else:
        image_stretch = gray_process(image)
    return image_stretch

def truncate_test_osu(image, max_out = 255, min_out = 0):
    def gray_process(gray):
        truncated_value = 1
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out 
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        if(max_out <= 255):
            gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        elif(max_out <= 65535):
            gray = cv2.normalize(gray, gray, 0, 65535, cv2.NORM_MINMAX)
        return (gray-gray.min())/(gray.max()-gray.min())
    
    #  如果是多波段
    if(len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    #  如果是单波段
    else:
        image_stretch = gray_process(image)
    return image_stretch

def truncate_train_ben(image, max_out = 255, min_out = 0):
    def gray_process(gray):
        truncated_value = random.uniform(0,3)
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out 
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        if(max_out <= 255):
            gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        elif(max_out <= 65535):
            gray = cv2.normalize(gray, gray, 0, 65535, cv2.NORM_MINMAX)
        return (gray-gray.min())/(gray.max()-gray.min())
    
    #  如果是多波段
    if(len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    #  如果是单波段
    else:
        image_stretch = gray_process(image)
    return image_stretch

def truncate_test_ben(image, max_out = 255, min_out = 0):
    def gray_process(gray):
        truncated_value = 1.5
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out 
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        if(max_out <= 255):
            gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        elif(max_out <= 65535):
            gray = cv2.normalize(gray, gray, 0, 65535, cv2.NORM_MINMAX)
        return (gray-gray.min())/(gray.max()-gray.min())
    
    #  如果是多波段
    if(len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    #  如果是单波段
    else:
        image_stretch = gray_process(image)
    return image_stretch