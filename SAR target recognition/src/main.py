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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim.lr_scheduler as lr_scheduler
import network
import sampler
import OpenSARship_dataset
import FuSARship_dataset
import MSTAR_dataset
import transform_data
from collections import OrderedDict
from models.build import build_model
from models.load_pretrain import load_pretrained_swinT,load_pretrained_tiny

def Model_prepare(Model_type,pretrain_path):
    if Model_type == 'ResNet18':
        model = network.ResNet18_opti_rs(tsx_num_class = 32)
        num_feature = model.fc.in_features
        checkpoint = torch.load(pretrain_path)  # 加载模型文件，pt, pth 文件都可以；
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        model.fc = nn.Linear(num_feature, 10)
    elif Model_type == 'ResNet50':
        model = network.ResNet50_opt(tsx_num_classes = 32)
        num_feature = model.fc.in_features
        checkpoint = torch.load(pretrain_path)  # 加载模型文件，pt, pth 文件都可以；
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        model.fc = nn.Linear(num_feature, 10)
    elif Model_type == 'ResNet101':
        model = network.ResNet101(tsx_num_classes = 32)
        num_feature = model.fc.in_features
        checkpoint = torch.load(pretrain_path)
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        model.fc = nn.Linear(num_feature, 10)
    elif Model_type == 'DenseNet121':
        model = network.densenet121(target_num_classes = 32)
        num_feature = model.classifier.in_features
        checkpoint = torch.load(pretrain_path)
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        model.classifier = nn.Linear(num_feature, 10)
    elif Model_type == 'SENet':
        model = network.Senet((3,4,6,3),num_classes = 32)
        num_feature = model.fc.in_features
        checkpoint = torch.load(pretrain_path)
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        model.fc = nn.Linear(num_feature, 6)
    elif Model_type == 'MobileV3':
        model = network.mobilenetv3_small()
        checkpoint = torch.load(pretrain_path)
        new_state_dict = checkpoint
        model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
        num_feature = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_feature, 10)
    elif Model_type == 'Swin-T':
        pretrain_is_official = 0
        model = build_model(mode = 'train',model_type= 'Swin-T-TSX')
        state_dict = load_pretrained_swinT(pretrain_path,model,pretrain_is_official)
        model.load_state_dict(state_dict,strict=False)
        num_feature = model.head.in_features
        del model.head
        model.head = nn.Linear(num_feature, 10)
    elif Model_type == 'Swin-B':
        pretrain_is_official = 0
        model = build_model(mode = 'train',model_type= 'Swin-B-TSX')
        state_dict = load_pretrained_swinT(pretrain_path,model,pretrain_is_official)
        model.load_state_dict(state_dict,strict=False)
        num_feature = model.head.in_features
        del model.head
        model.head = nn.Linear(num_feature, 10)
    model = model.cuda()
    return model

def Mag_Train(Model_type, tensorboard_save):
    normalize = transforms.Normalize(mean=[0.051745296], std=[0.059024353])
    train_transform = transforms.Compose([

        transform_data.Numpy2Tensor(3),
        normalize,
        transforms.CenterCrop(128),         ## only for MSTAR and FuSARShip
    ])
    val_transform = transforms.Compose([
        transform_data.Numpy2Tensor(3),
        normalize,
        transforms.CenterCrop(128),
    ])
    '''
    # MSTAR
    # 10% train
    train_list_path = 'data/MSTAR/train_10.txt'
    # 30% train
    # train_list_path = 'data/MSTAR/train_30.txt'
    # 100% train
    # train_list_path = 'data/MSTAR/train.txt'
    test_list_path = 'data/MSTAR/test.txt'
    train_dataset = MSTAR_dataset.MsTAR_Dataset(train_list_path, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32,shuffle=True,num_workers=8)
    val_dataset = MSTAR_dataset.MsTAR_Dataset(test_list_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=8)
    '''
    '''
    # OpenSARship
    # large scale
    train_list_path = 'data/OSS/large_train.txt'
    test_list_path = 'data/OSS/large_test.txt'
    # medium scale
    # train_list_path = 'data/OSS/medium_train.txt'
    # test_list_path = 'data/OSS/medium_test.txt'
    # low scale
    # train_list_path = 'data/OSS/low_train.txt'
    # test_list_path = 'data/OSS/low_test.txt'
    
    train_dataset = OpenSARship_dataset.OpenSARshipDataset(train_list_path, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32,sampler=sampler.ImbalancedDatasetSampler(train_dataset),shuffle=False,num_workers=8)
    val_dataset = OpenSARship_dataset.OpenSARshipDataset(test_list_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True,num_workers=8)
    
    '''
    # FuSARship
    train_list_path = 'data/FSS/FuSAR_train6.txt'
    test_list_path = 'data/FSS/FuSAR_test6.txt'
    
    train_dataset = FuSARship_dataset.FuSARshipDataset(train_list_path, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32,sampler=ImbalancedDatasetSampler(train_dataset),shuffle=False,num_workers=8)
    val_dataset = FuSARship_dataset.FuSARshipDataset(test_list_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True,num_workers=8)
    

    
    model = Model_prepare(Model_type,pretrain_path)
    
    # CNN
    optimizer = SGD(model.parameters(),
                0.01, momentum=0.9, weight_decay=1e-3, nesterov=True)
    lr_s = StepLR(optimizer, step_size=100, gamma=0.5)
    """
    # ViTs
    optimizer = torch.optim.AdamW(model.parameters(), eps = 1e-8, lr = 0.000005, weight_decay = 0.001)
    lr_s = StepLR(optimizer, step_size=100, gamma=0.1)
    """
# start training
    best_acc1 = 0.
    epochs=500
    writer = SummaryWriter(tensorboard_save)
    for epoch in range(epochs):
        # train for one epoch
        train_loss = train(train_dataloader, model, optimizer,lr_s,
             epoch)

        # evaluate on validation set
        if epoch>=0:
            acc1,val_loss = validate(val_loader, model)
            print('准确率：{}'.format(acc1.item()),'val_loss: {}'.format(val_loss.item()))
        # remember best acc@1 and save checkpoint
        # if acc1 > best_acc1:
        #     best_model = copy.deepcopy(model.state_dict())
            best_acc1 = max(acc1, best_acc1)
            lr_s.step()
            writer.add_scalars('loss', {'train': train_loss.item()}, epoch + 1)
            writer.add_scalars('val_acc', {'train': best_acc1}, epoch + 1)
            writer.add_scalars('val_loss',{'train': val_loss.item()},epoch+1)
    print('best_acc:{}'.format(best_acc1))
def train(train_dataloader, model, optimizer,scheduler,
             epoch):
    F = nn.CrossEntropyLoss()
    model.train()
    for idx,(t_data,t_target) in enumerate(train_dataloader):
        t_data,t_target=t_data.cuda(),t_target.cuda()
        pred = model(t_data)#batch_size*2
        loss = F(pred,t_target)

        #Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
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
            # print(i)
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

if __name__ == '__main__':
    Model_type = 'SENet'
    pretrain_path = 'pretrain_models/SENet50_TSX.pth'
    tensorboard_save = 'SENet_FuSARship_IMG'
    Mag_Train(Model_type,tensorboard_save,pretrain_path)
