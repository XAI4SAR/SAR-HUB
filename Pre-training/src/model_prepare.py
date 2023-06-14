from models.build import build_model
from models.swin_load_pretrain import load_pretrained_swinT
import torch.nn as nn
from collections import OrderedDict
import torch
from models import network
def load_pretrained_model(config):
    sto = config['if_sto']
    if config['Model_Type'] == 'Swin-T':
        transferred_model = build_model('train',config['Model_Type'])
        if not sto:
            state_dict = load_pretrained_swinT(config['models']['load_model_path'],transferred_model)
            transferred_model.load_state_dict(state_dict,strict=False)
        num_feature = transferred_model.head.in_features
        del transferred_model.head
        transferred_model.head = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'Swin-B':
        transferred_model = build_model('train',config['Model_Type'])
        if not sto:
            state_dict=load_pretrained_swinT(config['models']['load_model_path'], transferred_model)
            transferred_model.load_state_dict(state_dict,strict=False)
        num_feature = transferred_model.head.in_features
        del transferred_model.head
        transferred_model.head = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'MobileV3':
        transferred_model = network.mobilenetv3_small()
        if not sto:
            checkpoint = torch.load(config['models']['load_model_path'], map_location='cpu') 
            transferred_model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
        num_feature = transferred_model.classifier[3].in_features
        transferred_model.classifier[3] = nn.Linear(num_feature, config['cate_num'])
        
    elif config['Model_Type'] == 'ResNet50':
        transferred_model = network.ResNet50(51)
        num_feature = transferred_model.fc.in_features
        if not sto:
            new_state_dict = OrderedDict()
            checkpoint = torch.load(config['models']['load_model_path'], map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
            new_state_dict = checkpoint['model']
            transferred_model.load_state_dict(new_state_dict)
        transferred_model.fc = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'ResNet18':
        transferred_model = network.ResNet18_TSX(45)
        num_feature = transferred_model.fc.in_features
        if not sto:
            new_state_dict = OrderedDict()
            checkpoint = torch.load(config['models']['load_model_path'], map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
            new_state_dict = checkpoint
            transferred_model.load_state_dict(new_state_dict)
        transferred_model.fc = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'ResNet101':
        transferred_model = network.ResNet101(45)
        num_feature = transferred_model.fc.in_features
        if not sto:
            new_state_dict = OrderedDict()
            checkpoint = torch.load(config['models']['load_model_path'], map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
            new_state_dict = checkpoint
            transferred_model.load_state_dict(new_state_dict)
        transferred_model.fc = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'DenseNet121':
        transferred_model = network.densenet121(target_num_classes = 45)
        num_feature = transferred_model.classifier.in_features
        if not sto:
            new_state_dict = OrderedDict()
            checkpoint = torch.load(config['models']['load_model_path'], map_location='cpu')  
            new_state_dict = checkpoint
            transferred_model.load_state_dict(new_state_dict)
        transferred_model.classifier = nn.Linear(num_feature, config['cate_num'])
    elif config['Model_Type'] == 'SENet50':
        transferred_model = network.Senet((3,4,6,3),45)
        num_feature = transferred_model.fc.in_features
        if not sto:
            new_state_dict = OrderedDict()
            checkpoint = torch.load(config['models']['load_model_path'], map_location='cpu')  # 加载模型文
            new_state_dict = checkpoint
            transferred_model.load_state_dict(new_state_dict)
        transferred_model.fc = nn.Linear(num_feature, config['cate_num'])
    else:
        print('No Model Type Config!')
    model = transferred_model.cuda()
    return model, config['Model_Type']
