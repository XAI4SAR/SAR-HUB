
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
import argparse
from models.swin_load_pretrain import load_pretrained_swinT
def model_preparing(config):
    if config['Model_Type'] == 'Swin-T':
        transferred_model = build_model('train',config['Model_Type'])
        state_dict = load_pretrained_swinT(config['models'],transferred_model)
        transferred_model.load_state_dict(state_dict,strict=False)

    elif config['Model_Type'] == 'Swin-B':
        transferred_model = build_model('train',config['Model_Type'])
        state_dict=load_pretrained_swinT(config['models'], transferred_model)
        transferred_model.load_state_dict(state_dict,strict=False)
    elif config['Model_Type'] == 'MobileV3':
        transferred_model = network.mobilenetv3_small()
        checkpoint = torch.load(config['models'], map_location='cpu') 
        transferred_model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
        
    elif config['Model_Type'] == 'ResNet50':
        
        transferred_model = network.ResNet50(51)
        checkpoint = torch.load(config['models'], map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
        transferred_model.load_state_dict(new_state_dict)
    elif config['Model_Type'] == 'ResNet18':
        
        transferred_model = network.ResNet18_TSX(45)
        checkpoint = torch.load(config['models'], map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
        new_state_dict = checkpoint
        transferred_model.load_state_dict(new_state_dict)
    elif config['Model_Type'] == 'ResNet101':
        
        transferred_model = network.ResNet101(45)
        checkpoint = torch.load(config['models'], map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
        transferred_model.load_state_dict(new_state_dict)
    elif config['Model_Type'] == 'DenseNet121':
        
        transferred_model = network.densenet121(target_num_classes = 45)
        checkpoint = torch.load(config['models'], map_location='cpu')  
        new_state_dict = checkpoint
        transferred_model.load_state_dict(new_state_dict)
    elif config['Model_Type'] == 'SENet50':
        transferred_model = network.Senet((3,4,6,3),45)
        checkpoint = torch.load(config['models'], map_location='cpu')  
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
                data_transform.Reinhard_test(config['para']),
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
            data_transform.truncated_test_ben(config['para']),
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
            data_transform.truncated_test_osu(config['para']),
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

    precision = [matrix[i, i] / sum(matrix[:, i]) for i in range(label_num)] 
    recall = [matrix[i, i] / sum(matrix[i, :]) for i in range(label_num)] 
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
def parameter_setting(args):
    config = {}

    config['Model_Type'] = args.model    
    config['models'] = args.pretrained_path
    config['dataset'] = args.dataset 
    config['DRAE'] = args.DRAE
    config['batch_size'] = args.batch 
    
    if config['dataset'] == 'TerraSAR-X':
        if config['DRAE'] == 'Reinhard':
            config['datatxt_test'] = 'data/tsx_test.txt'
            config['nor_mean'] = 0.17721633016340846
            config['nor_std'] = 0.023696591996910408
            config['cate_num'] = 32
            config['para'] = [3.5,4.5]
        else:
            raise NameError('Non-corresponding DRAE functions and dataset.')
        
    elif config['dataset'] == 'BigEarthNet-Small':
        if config['DRAE'] == 'PTLS':
            config['datatxt_test'] = 'data/ben_test.txt'
            config['nor_mean'] = 0.5995
            config['nor_std'] = 0.0005743462
            config['cate_num'] = 19
            config['para'] = [0,2]
        else:
            raise NameError('Non-corresponding DRAE functions and dataset.')
        
    elif config['dataset'] == 'OpenSARUrban':
        if config['DRAE'] == 'PTLS':
            config['datatxt_test'] = 'data/ben_test.txt'
            config['nor_mean'] = 0.3159206957121415
            config['nor_std'] = 0.034312685984107194
            config['cate_num'] = 10
            config['para'] = [0,3]
        else:
            raise NameError('Non-corres ponding DRAE functions and dataset.')
                   
    return config
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Upstream_Training')
    
    parser.add_argument('--pretrained_path', default='ResNet18_TSX.pth', help='Path of the SAR pre-trained backbone path.')
    parser.add_argument('--model', type=str, default='ResNet18', help='The model type in inference time.')
    # Optional: ResNet18 ResNet50 DenseNet121 SENet50 MobileV3 Swin-T Swin-B
    parser.add_argument('--dataset', type=str, default='TerraSAR-X', help='The dataset in training process.')
    # Optional: TerraSAR-X BigEarthNet-Small OpenSARUrban
    parser.add_argument('--batch', type=int, default=200, help='The instances number of each batch during inference.')
    parser.add_argument('--DRAE', type=str, default='PTLS', help='The DRAE function used in the inference.')
    # PTLS Reinhard
    args = parser.parse_args([])
    config = parameter_setting(args)
    image_test(config)