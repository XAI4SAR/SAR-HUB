
import torch
import os
import dataset
import data_transform
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model_save_load import save_model_DDP
import torch.nn as nn
from torch.optim import lr_scheduler
from loss import  CB_Focal_Loss, MINI_CB_Focal_Loss
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from model_prepare import load_pretrained_model
from dist_val import image_val_BEN,image_val_TSX_OSU
import numpy as np 
def get_dataloader(config):

    nor_mean = config['nor_mean']
    nor_std = config['nor_std']
    if config['dataset'] == 'TerraSAR-X':
        #! TerraSAR-X、BigEarthNet-Small or OpenSARUrban
        data_transforms = {
            'train': transforms.Compose([
                data_transform.Reinhard_train(config['para']),
                data_transform.Normalize(nor_mean,nor_std),
                
                data_transform.Numpy2Tensor(3),
                transforms.Resize(128),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
            ]),
            'val': transforms.Compose([
                data_transform.Reinhard_test(config['para']),
                data_transform.Normalize(nor_mean,nor_std),
                data_transform.Numpy2Tensor(3),
                transforms.Resize(128),
            ]),}
        dataset_train = dataset.TSX_Dataset(txt_file=config['datatxt_train'],
                                    root_dir=config['data_root'],
                                    data_type='npy' ,
                                    transform=data_transforms['train'],
                                    )
        dataset_val = dataset.TSX_Dataset(txt_file=config['datatxt_val'],
                                root_dir=config['data_root'],
                                data_type='npy',
                                
                                transform=data_transforms['val'],
                                )
    elif config['dataset'] == 'BigEarthNet-Small':
        data_transforms = {
        'train': transforms.Compose([
            data_transform.truncated_train_ben(config['para']),
            data_transform.Normalize(nor_mean,nor_std),
            data_transform.Numpy2Tensor(3),
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
        ]),
        'val': transforms.Compose([
            data_transform.truncated_test_ben(config['para']),
            data_transform.Normalize(nor_mean,nor_std),
            data_transform.Numpy2Tensor(3),
            transforms.Resize(128),
        ]),

        }

        dataset_train = dataset.BEN_Dataset(txt_file=config['datatxt_train'],
                                    root_dir=config['data_root'],
                                    data_type='npy' ,
                                    transform=data_transforms['train'],
                                    )
        dataset_val = dataset.BEN_Dataset(txt_file=config['datatxt_val'],
                                root_dir=config['data_root'],
                                data_type='npy',
                                transform=data_transforms['val'],
                                )
    elif config['dataset'] == 'OpenSARUrban':
        data_transforms = {
        'train': transforms.Compose([
            data_transform.truncated_train_osu(config['para']),
            data_transform.Normalize(nor_mean,nor_std),
            data_transform.Numpy2Tensor(3),
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
        ]),
        'val': transforms.Compose([
            data_transform.truncated_test_osu(config['para']),
            data_transform.Normalize(nor_mean,nor_std),
            data_transform.Numpy2Tensor(3),
            transforms.Resize(128),
        ]),
        }
        dataset_train = dataset.OSU_Dataset(
                                txt_file=config['datatxt_train'],
                                root_dir=config['data_root'],
                                data_type='npy' ,
                                transform=data_transforms['train']
                                )
        dataset_val = dataset.OSU_Dataset(
                                txt_file=config['datatxt_val'],
                                root_dir=config['data_root'],
                                data_type='npy',
                                transform=data_transforms['val']
                                )
    else:
        raise NameError('Unknown Dataset Type')
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle = True)
    dataloader = {}
    dataloader['train'] = DataLoader(dataset_train,
                        batch_size=config['batch_size'][0],
                        shuffle=False,
                        sampler=train_sampler,
                        num_workers=8,
                        pin_memory=True,
                        )
    dataloader['val'] = DataLoader(dataset_val,
                        batch_size=config['batch_size'][1],
                        shuffle=True,
                        
                        num_workers=8,
                        pin_memory=True,
                        )
    return dataloader,  train_sampler
def reduce_value(value, world_size, average=True):
    world_size = world_size
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value) # 求和
        if average:  # 取平均
            value /= world_size

        return value
    

def train_loss(config):
    if config['dataset'] == 'TerraSAR-X':
        #! TerraSAR-X、BigEarthNet-Small or OpenSARUrban
        cls_num_list = [23181, 1524, 96, 85, 966, 129, 2270, 531, 71, 3376, 399, 129, 599, 1091, 3018, 21, 580, 706, 42, 447, 354, 693, 581, 1682, 1675, 1272, 83, 164, 58, 223, 163, 191]
    elif config['dataset'] == 'BigEarthNet-Small':
        cls_num_list = [6153, 1732, 15493, 2213, 7835, 8023, 11012, 2281, 11365, 15940, 16285, 1070, 1572, 13225, 125, 2264, 194, 7006, 6235]
    elif config['dataset'] == 'OpenSARUrban':
        cls_num_list = [3152, 2660, 2161, 1138, 447, 652, 1061, 135, 64, 33]
    else:
        raise TypeError('Unknown Dataset Type')
    if  config['loss_type'] == 'CB_Focal':
            criterion = CB_Focal_Loss(samples_per_cls=cls_num_list,beta = 0.995, num_of_cls=config['cate_num'], gamma = 2)
    elif config['loss_type'] == 'Mini_CB_FL':
            criterion = MINI_CB_Focal_Loss(samples_per_cls=cls_num_list,beta = 0.995, num_of_cls=config['cate_num'], gamma = 2)
    else:
        raise TypeError('Unknown Loss Type')
    return criterion, sum(cls_num_list)

def resnet_train(config,world_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,type = load_pretrained_model(config)
    dataloader, train_sampler=get_dataloader(config,type)
    criterion, number_of_dataset = train_loss(config)
    if type == 'Swin-T' or type == 'Swin-B':
        optimizer = torch.optim.AdamW(model.parameters(), eps = 1e-8, lr = 0.000005, weight_decay = 0.05)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr = 0.000001*25,steps_per_epoch=2*(int(number_of_dataset/(config['batch_size'][0]*1))+1), epochs=150,anneal_strategy='cos')
    else:
        optimizer = torch.optim.AdamW(model.parameters(), eps = 1e-8, lr = 0.00005, weight_decay = 0.05)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr = 0.00001*25,steps_per_epoch=2*(int(number_of_dataset/(config['batch_size'][0]*1))+1), epochs=150,anneal_strategy='cos')
    
    model.to(device)
    i = 0 
    for epoch in range(config['num_epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, data in enumerate(dataloader['train']):
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #! 使用OneCycleLr，学习率需要在每一个batch迭代更新
            scheduler.step()
            loss = reduce_value(loss, world_size, average=True)
            i += 1
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        if config['dataset'] == 'TerraSAR-X' or config['dataset'] == 'OpenSARUrban':
            val_accuracy,val_loss = image_val_TSX_OSU(dataloader['val'], device,model,world_size,training = False)
            
        elif config['dataset'] == 'BigEarthNet-Small':
            val_acc = 0
            val_top_acc,prec,recall, val_loss = image_val_BEN(dataloader['val'], device,model,world_size,training = False)
            val_acc1 = val_top_acc['f1_micro']
            val_accuracy = np.hstack((val_acc, val_acc1))
            
        if dist.get_rank() == 0:
            print(' training epoch:', epoch + 1, 'iter: ', str(i + 1), 'ce_loss: ', loss,'\n')
            print('val accuracy:', val_accuracy,', val_loss:', val_loss,'\n')
            
        if dist.get_rank() == 0:
            save_model_DDP(model, config['models']['save_model_path'] + 'epoch' + str(epoch + 1) + '.pth')
            print('save model... epoch' + str(epoch + 1))
