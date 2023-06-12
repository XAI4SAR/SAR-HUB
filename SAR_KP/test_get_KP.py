import os
import numpy as np
import torch.utils.data.dataloader as DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.optim import SGD
import unet
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim.lr_scheduler as lr_scheduler
import resnet
import MSTARdataset
from collections import OrderedDict
import transform_data
import cv2
from skimage import io, transform
import skimage

def Model_resnet(type):
    if type == 'OPT':
        path = 'resnet50_OPT_10_model/epoch.pth'
        num_classes = 10
    elif type == 'IMG':
        path = 'resnet50_IMG_10_model/epoch.pth'
        num_classes = 10
    elif type == 'SAR':
        path = 'resnet50_OSU_10_model/epoch.pth'
        num_classes = 10
    else:
        raise TypeError('ResNet50 Model Type Error')
    model = resnet.ResNet50(num_classes = num_classes)
    num_feature = model.fc.in_features
    checkpoint = torch.load(path)  
    new_state_dict = checkpoint
    model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
    model.fc = nn.Linear(num_feature, 10)
    model = model.cuda()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    return model
def Model_unet(sample_time,type):
    model = unet.U_Net(sample_time)
    if type == 'OPT':
        path = 'optical/100.pth'
    elif type == 'IMG':
        path = 'ImageNet/100.pth'
    elif type == 'SAR':
        path = 'SAR/100.pth'
    else:
        raise TypeError('UNET Model Type Error')
    checkpoint = torch.load(path,map_location = 'cpu')
    model.load_state_dict(checkpoint)
    model = model.cuda()
    return model
def kownledge_point_loss(input_feature,noise_feature,noise,batch,alpha):
    feature_diff = (((noise_feature-input_feature)**2).sum())/batch
    entropy = alpha*torch.sum(noise)
    delta_f = torch.var(input_feature)
    loss = (1/delta_f)*feature_diff-entropy
    return loss,entropy,feature_diff
def dataset(transform,img_list_path):
    test_list_path = img_list_path
    test_dataset = MSTARdataset.MsTAR_Dataset(test_list_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=True,num_workers=8)
    return test_dataset,test_dataloader
        
def test_list(normalize,transform,alpha,img_list_path,type):
    test_dataset,test_dataloader = dataset(transform,img_list_path)
    unet = Model_unet(15,type)
    resnet_model = Model_resnet(type)
    unet.eval()
    resnet_model.eval()
    for idx,(t_data,t_target) in enumerate(test_dataloader):
        true_data,true_target=t_data[0].cuda(),t_target.cuda()
        img_path = t_data[1][0]
        input_noise,noise= unet(true_data)
        input_noise_norm = normalize(input_noise)
        true_data_norm = normalize(true_data)
        _,true_data_feature_layer4 = resnet_model(true_data_norm)
        _,input_noise_feature_layer4 = resnet_model(input_noise_norm)
        loss,entropy,feature_diff = kownledge_point_loss(true_data_feature_layer4,input_noise_feature_layer4,noise,1,alpha)
        n = torch.mean(noise,dim=0).cpu().detach().numpy()
        n = ((n-n.min())/(n.max()-n.min()))*255
        n = skimage.transform.resize(n, (128, 128))
        noise = noise.cpu().detach().numpy()
        np.save(img_path.split('/')[-1][:-4]+'+'+type+'_last_model_deltaX.npy',noise)


def test_single_img(normalize,transform,alpha,img_path,type):
    unet = Model_unet(15,type)
    resnet_model = Model_resnet(type)
    unet.eval()
    resnet_model.eval()
    img = np.load(img_path)
    img_trans = transform(img)
    true_data = img_trans.cuda()
    true_data = true_data.repeat(1,1,1,1)
    input_noise,noise= unet(true_data)
    input_noise_norm = normalize(input_noise)
    true_data_norm = normalize(true_data)
    _,true_data_feature_layer4 = resnet_model(true_data_norm)
    _,input_noise_feature_layer4 = resnet_model(input_noise_norm)
    loss,entropy,feature_diff = kownledge_point_loss(true_data_feature_layer4,input_noise_feature_layer4,noise,1,alpha)
    n = torch.mean(noise,dim=0).cpu().detach().numpy()
    n = ((n-n.min())/(n.max()-n.min()))*255
    n = skimage.transform.resize(n, (128, 128))
    noise = noise.cpu().detach().numpy()
    cv2.imwrite(img_path.split('/')[-1][:-4]+'+'+type+'_last_model_deltaX.jpg',n)
    np.save(img_path.split('/')[-1][:-4]+'+'+type+'_last_model_deltaX.npy',noise)

def mag_test(img_path,img_list_path,test_single,type):

    normalize = transforms.Normalize(mean=[0.051745296], std=[0.059024353])
    alpha = 0.25
    transform = transforms.Compose([
      
        transform_data.Numpy2Tensor(3),
        transforms.CenterCrop(128),

    ])
    if test_single:
        test_single_img(normalize,transform,alpha,img_path,type)
    else:
        test_list(normalize,transform,alpha,img_list_path,type)
    
if __name__ == '__main__':
    test_single =0
    for type in ['IMG','OPT','SAR']:
        img_path = 'MSTAR/SOC/test/BMP-2/HB03335.000_Mag.npy'
        img_list_path = 'src/all.txt'
        mag_test(img_path,img_list_path,test_single,type)