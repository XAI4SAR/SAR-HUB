
import torch
import dataset
import data_transform
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, dataloader
from models import network
import torch.nn as nn
import argparse
import numpy as np
import datetime
from models.build import build_model
from models.swin_load_pretrain import load_pretrained_swinT
def model_preparing(config):
    if config['Backbone'] == 'Swin-T':
        transferred_model = build_model('train',config['Backbone'])
        state_dict = load_pretrained_swinT('model/Swin-T_epoch_300.pth',transferred_model)
        transferred_model.load_state_dict(state_dict,strict=False)

    elif config['Backbone'] == 'Swin-B':
        transferred_model = build_model('train',config['Backbone'])
        state_dict=load_pretrained_swinT('model/Swin-B_epoch_300.pth', transferred_model)
        transferred_model.load_state_dict(state_dict,strict=False)
    elif config['Backbone'] == 'MobileV3':
        transferred_model = network.mobilenetv3_small()
        checkpoint = torch.load('model/Mobilev3_epoch_300.pth', map_location='cpu') 
        transferred_model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
        
    elif config['Backbone'] == 'ResNet50':
        
        transferred_model = network.ResNet50(51)
        checkpoint = torch.load('model/ResNet50_epoch_300.pth', map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
        transferred_model.load_state_dict(new_state_dict)
    elif config['Backbone'] == 'ResNet18':
        
        transferred_model = network.ResNet18_TSX(45)
        checkpoint = torch.load('model/ResNet18_epoch_300.pth', map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
        new_state_dict = checkpoint
        transferred_model.load_state_dict(new_state_dict)
    elif config['Backbone'] == 'ResNet101':
        
        transferred_model = network.ResNet101(45)
        checkpoint = torch.load('model/resnet101_epoch_300.pth', map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
        transferred_model.load_state_dict(new_state_dict)
    elif config['Backbone'] == 'DenseNet121':
        
        transferred_model = network.densenet121(target_num_classes = 45)
        checkpoint = torch.load('model/densenet121_epoch_300.pth', map_location='cpu')  
        new_state_dict = checkpoint
        transferred_model.load_state_dict(new_state_dict)
    elif config['Backbone'] == 'SENet50':
        transferred_model = network.Senet((3,4,6,3),45)
        checkpoint = torch.load('model/Senet50_epoch_300.pth', map_location='cpu')  # 加载模型文
        new_state_dict = checkpoint
        transferred_model.load_state_dict(new_state_dict)
    else:
        print('No Model Type Config!')
    return transferred_model.cuda()

def data_preparing(config):
    '''
    data preparing
    '''
    nor_mean = config['nor_mean']
    nor_std = config['nor_std']
    if config['dataset'] == 'TerraSAR-X':
        #! TerraSAR-X、BigEarthNet-Small or OpenSARUrban
        data_transforms = {
            'val': transforms.Compose([
                data_transform.Reinhard_test(),
                data_transform.Normalize(nor_mean,nor_std),
                data_transform.Numpy2Tensor(3),
                transforms.Resize(128),
            ]),}
        dataset_test = dataset.TSX_Dataset(txt_file=config['datatxt_test'],
                                root_dir=config['data_root'],
                                data_type='npy',
                                label_smooth=0,
                                transform=data_transforms,
                                )
    elif config['dataset'] == 'BigEarthNet-Small':
        data_transforms = {
        transforms.Compose([
            data_transform.truncated_test_ben(),
            data_transform.Normalize(nor_mean,nor_std),
            data_transform.Numpy2Tensor(3),
            transforms.Resize(128),
        ]),

        }
        dataset_test = dataset.BEN_Dataset(txt_file=config['datatxt_test'],
                                root_dir=config['data_root'],
                                data_type='npy',
                                transform=data_transforms,
                                )
    elif config['dataset'] == 'OpenSARUrban':
        data_transforms = {
            transforms.Compose([
            data_transform.truncated_test_osu(),
            data_transform.Normalize(nor_mean,nor_std),
            data_transform.Numpy2Tensor(3),
            transforms.Resize(128),
        ]),
        }
        dataset_test = dataset.OSU_Dataset(
                                txt_file=config['datatxt_test'],
                                root_dir=config['data_root'],
                                data_type='npy',
                                transform=data_transforms
                                )
    dataloader_test = DataLoader(dataset_test,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)
    return dataloader_test

def result_evaluation(matrix):

    label_num = matrix.shape[0]

    precision = [matrix[i, i] / sum(matrix[:, i]) for i in range(label_num)]  ##查全率
    recall = [matrix[i, i] / sum(matrix[i, :]) for i in range(label_num)]  ## 查准率
    f1_score = [2.0 / (1.0 / precision[i] + 1.0 / recall[i]) for i in range(label_num)] ## F1
    
    return precision, recall, f1_score

def image_test(config):
    dataloader = data_preparing(config)
    
    model = model_preparing(config)
    acc_num = 0.0
    data_num = 0.0
    output_softmax = nn.Softmax()
    device = config['device']
    model.to(device)
    model.eval()
    iter_test = iter(dataloader)
    for i in range(len(dataloader)):
        data = iter_test.next()
        inputs = data['image']
        labels = data['label']
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        outputs = output_softmax(out)

        _, predict = torch.max(outputs,1)
        acc_num += torch.sum(torch.squeeze(predict) == labels.data).float()
        data_num += labels.size()[0]
    test_acc = acc_num / data_num
    print(test_acc)
def parameter_setting():
    config = {}

    config['Backbone'] = ' '     # Model Type: ResNet-18,ResNet-50,ResNet-101,
    config['dataset'] = 'TerraSAR-X' #! TerraSAR-X、BigEarthNet-Small or OpenSARUrban
    config['batch_size'] = 10
    
    if config['dataset'] == 'TerraSAR-X':
        config['datatxt_test'] = 'data/tsx_test.txt'
        config['nor_mean'] = 0.17721633016340846
        config['nor_std'] = 0.023696591996910408
        config['cate_num'] = 32
        
    elif config['dataset'] == 'BigEarthNet-Small':
        config['datatxt_test'] = 'data/ben_test.txt'
        config['nor_mean'] = 0.5995
        config['nor_std'] = 0.0005743462
        config['cate_num'] = 19
        
    elif config['dataset'] == 'OpenSARUrban':
        config['datatxt_test'] = 'data/osu_test.txt'
        config['nor_mean'] = 0.3159206957121415
        config['nor_std'] = 0.034312685984107194
        config['cate_num'] = 10                     
    return config
if __name__ == '__main__':
    config = parameter_setting()s
    image_test(config)