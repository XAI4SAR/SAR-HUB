import os
import numpy as np
import torch.utils.data.dataloader as DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.optim import SGD
from sampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from models import network
import sampler
import datasets
import transform_data
from collections import OrderedDict
from models.build import build_model
from models.load_pretrain import load_pretrained_swin
import argparse

def Model_prepare(config):
    if config['Model_Type'] == 'ResNet18':
        model = network.ResNet18_opti_rs(tsx_num_class = config['pre_class'])
        num_feature = model.fc.in_features
        checkpoint = torch.load(config['models'])  
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        model.fc = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'ResNet50':
        model = network.ResNet50_opt(tsx_num_classes = config['pre_class'])
        num_feature = model.fc.in_features
        checkpoint = torch.load(config['models'])  
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        model.fc = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'ResNet101':
        model = network.ResNet101(tsx_num_classes = config['pre_class'])
        num_feature = model.fc.in_features
        checkpoint = torch.load(config['models'])
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        model.fc = nn.Linear(num_feature,config['cate_num'])
    elif config['Model_Type'] == 'DenseNet121':
        model = network.densenet121(target_num_classes = config['pre_class'])
        num_feature = model.classifier.in_features
        checkpoint = torch.load(config['models'])
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        model.classifier = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'SENet':
        model = network.Senet((3,4,6,3),num_classes = config['pre_class'])
        num_feature = model.fc.in_features
        checkpoint = torch.load(config['models'])
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        model.fc = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'MobileV3':
        model = network.mobilenetv3_small(num=config['pre_class'])
        checkpoint = torch.load(config['models'])
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        num_feature = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'Swin-T':
        model = build_model(model_type = config['Model_Type'],pre_class=config['pre_class'])
        state_dict = load_pretrained_swin(config['models'],model)
        model.load_state_dict(state_dict,strict=False)
        num_feature = model.head.in_features
        del model.head
        model.head = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'Swin-B':
        model = build_model(model_type = config['Model_Type'],pre_class=config['pre_class'])
        state_dict = load_pretrained_swin(config['models'],model)
        model.load_state_dict(state_dict,strict=False)
        num_feature = model.head.in_features
        del model.head
        model.head = nn.Linear(num_feature, config['cate_num'])
    model = model.cuda()
    return model

def Mag_Train(config):
    
    if config['dataset'] == 'MSTAR':
        if config['dataset_sub'] == 'MSTAR_10':
            train_list_path = 'data/MSTAR/train_10.txt'
        elif config['dataset_sub'] == 'MSTAR_30':
            train_list_path = 'data/MSTAR/train_30.txt'
        elif config['dataset_sub'] == 'MSTAR_100':
            train_list_path = 'data/MSTAR/train.txt'
        else:
            raise NameError('Non-corresponding dataset.')
        normalize = transforms.Normalize(mean=[0.051745296], std=[0.059024353])
        train_transform = transforms.Compose([
            transform_data.Numpy2Tensor(3),
            normalize,
            transforms.CenterCrop(128), 
        ])
        val_transform = transforms.Compose([
            transform_data.Numpy2Tensor(3),
            normalize,
            transforms.CenterCrop(128),
        ])
        test_list_path = 'data/MSTAR/test.txt'
        train_dataset = datasets.MsTAR_Dataset(train_list_path, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=32,shuffle=True,num_workers=8)
        val_dataset = datasets.MsTAR_Dataset(test_list_path, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=8)
        
    elif config['dataset'] == 'OpenSARship':
        if config['dataset_sub'] == 'OpenSARship_low':
            train_list_path = 'data/OSS/low_train.txt'
            test_list_path = 'data/OSS/low_test.txt'
            normalize = transforms.Normalize(mean=[0.00307491], std=[0.00376823])
            train_transform = transforms.Compose([        
                transform_data.Numpy2Tensor(3),
                normalize,
                transforms.Resize(32),
            ])
            val_transform = transforms.Compose([
                transform_data.Numpy2Tensor(3),
                normalize,
                transforms.Resize(32),

            ])
        elif config['dataset_sub'] == 'OpenSARship_medium':
            train_list_path = 'data/OSS/medium_train.txt'
            test_list_path = 'data/OSS/medium_test.txt'
            normalize = transforms.Normalize(mean=[0.00184448], std=[0.00251385])
            train_transform = transforms.Compose([        
                transform_data.Numpy2Tensor(3),
                normalize,
                transforms.Resize(96),
            ])
            val_transform = transforms.Compose([
                transform_data.Numpy2Tensor(3),
                normalize,
                transforms.Resize(96),

            ])
        elif config['dataset_sub'] == 'OpenSARship_large':
            train_list_path = 'data/OSS/large_train.txt'
            test_list_path = 'data/OSS/large_test.txt'
            normalize = transforms.Normalize(mean=[0.00258246], std=[0.0032765])
            train_transform = transforms.Compose([        
                transform_data.Numpy2Tensor(3),
                normalize,
                transforms.Resize(128),
            ])
            val_transform = transforms.Compose([
                transform_data.Numpy2Tensor(3),
                normalize,
                transforms.Resize(128),

            ])
        else:
            raise NameError('Non-corresponding dataset.')
        train_dataset = datasets.OpenSARshipDataset(train_list_path, transform=train_transform,size = config['dataset_sub'])
        train_dataloader = DataLoader(train_dataset, batch_size=32,sampler=sampler.ImbalancedDatasetSampler(train_dataset),shuffle=False,num_workers=8)
        val_dataset = datasets.OpenSARshipDataset(test_list_path, transform=val_transform,size = config['dataset_sub'])
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True,num_workers=8)
    elif config['dataset'] == 'FuSARship':
        train_list_path = 'data/FSS/FuSAR_train6.txt'
        test_list_path = 'data/FSS/FuSAR_test6.txt'
        normalize = transforms.Normalize(mean=[0.0442077], std=[0.00497706])
        train_transform = transforms.Compose([
            transform_data.Numpy2Tensor(3),
            normalize,
            transforms.CenterCrop(128),
        ])
        val_transform = transforms.Compose([
            transform_data.Numpy2Tensor(3),
            normalize,
            transforms.CenterCrop(128),
        ])
        train_dataset = datasets.FuSARshipDataset(train_list_path, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=32,sampler=ImbalancedDatasetSampler(train_dataset),shuffle=False,num_workers=8)
        val_dataset = datasets.FuSARshipDataset(test_list_path, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True,num_workers=8)
    
    model = Model_prepare(config)
    
    if config['Model_Type'] == 'Swin-T' or config['Model_Type'] == 'Swin-B':
        optimizer = torch.optim.AdamW(model.parameters(), eps = 1e-8, lr = 0.0005, weight_decay = 0.0001)
        lr_s = StepLR(optimizer, step_size=100, gamma=0.5)
    else:
        optimizer = SGD(model.parameters(),
                0.01, momentum=0.9, weight_decay=1e-3, nesterov=True)
        lr_s = StepLR(optimizer, step_size=100, gamma=0.5)


# start training
    best_acc1 = 0.
    epochs=500
    for epoch in range(epochs):
        # train for one epoch
        train_loss = train(train_dataloader, model, optimizer,lr_s,
             epoch)
        # evaluate on validation set
        if epoch>=0:
            acc1,val_loss = validate(val_loader, model)
            print('准确率：{}'.format(acc1.item()),'val_loss: {}'.format(val_loss.item()))
            best_acc1 = max(acc1, best_acc1)
            lr_s.step()
    print('best_acc:{}'.format(best_acc1))

def train(train_dataloader, model, optimizer,scheduler,
             epoch):
    F = nn.CrossEntropyLoss()
    model.train()
    for idx,(t_data,t_target) in enumerate(train_dataloader):
        t_data,t_target=t_data.cuda(),t_target.cuda()
        pred = model(t_data)
        loss = F(pred,t_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch:{}, loss:{}".format(epoch,loss.item()))
    return loss

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])
        return correct
def validate(val_loader, model):
    F = nn.CrossEntropyLoss()
    # switch to evaluate mode
    model.eval()
    sum = 0
    with torch.no_grad():
        
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = F(output,target)

            # measure accuracy and record loss
            result = accuracy(output, target)
            sum += result
        acc = sum/len(val_loader)    
    return acc,loss

def parameter_setting(args):
    config = {}

    config['Model_Type'] = args.model    
    config['models'] = args.pretrained_path
    config['dataset'] = args.dataset 
    config['dataset_sub'] = args.dataset_sub
    config['pre_class'] = args.pre_class
    
    if config['dataset'] == 'MSTAR':
        config['cate_num'] = 10
    elif config['dataset'] == 'OpenSARship':
        if config['dataset_sub'] == 'OpenSARship_low':
            config['cate_num'] = 14
        elif config['dataset_sub'] == 'OpenSARship_medium':
            config['cate_num'] = 13
        elif config['dataset_sub'] == 'OpenSARship_large':
            config['cate_num'] = 7
        else:
            raise NameError('Non-corresponding dataset.')
    elif config['dataset'] == 'FuSARship':
        config['cate_num'] = 6
    else:
        raise NameError('Non-corresponding dataset.')
                   
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Target_recognition')
    
    parser.add_argument('--model', type=str, default='ResNet50', help='The model type.')
    # Optional: ResNet18 ResNet50 DenseNet121 SENet50 MobileV3 Swin-T Swin-B
    parser.add_argument('--dataset', type=str, default='OpenSARship', help='The dataset in training process.')
    # Optional: MSTAR OpenSARship FuSARship
    parser.add_argument('--dataset_sub', type=str, default='OpenSARship_medium', help='The sub dataset in training process.')
    # Optional: MSTAR_10 MSTAR_30 MSTAR_100 OpenSARship_low OpenSARship_medium OpenSARship_large FuSARship
    parser.add_argument('--pre_class', type=str, default=32, help='The category numbers of upstream datasets.')
    # Optional: ImageNet: 1000  Million-AID: 51 Nwpu-resisc45: 45 TSX: 32 BEN: 19 OSU:10
    parser.add_argument('--pretrained_path', type=str, default='pretrain_models/SENet50_TSX.pth', help='The SAR pre-treined model path.')
    
    args = parser.parse_args([])
    config = parameter_setting(args)
    Mag_Train(config)