import torch
import numpy as np
from torchvision import transforms
import math
from math import log
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random



def Reinhard_Devlin_train(img):
    amp_img = img
    u = 0.168605
    l = 0.0001              
    b = random.uniform(3.5, 4.5)
    img_a = l * amp_img + (1 - l) * u
    m = 0.3 + 0.7 * np.power(((1 - u) / (1 - np.min(amp_img))), 1.4)
    img_d = (amp_img / (amp_img + np.power((b * img_a), m)))
    return img_d

def Reinhard_Devlin_test(img):
    amp_img = img
    u = 0.169
    l = 0.0001              
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
    if(len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch
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
    
    if(len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
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
    
    if(len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
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
    if(len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    else:
        image_stretch = gray_process(image)
    return image_stretch