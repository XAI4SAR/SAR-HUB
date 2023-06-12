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
import dataset
from collections import OrderedDict
import transform_data
import cv2
from skimage import io, transform
def Model_resnet():
    model = resnet.ResNet50(num_classes = 32)
    num_feature = model.fc.in_features
    checkpoint = torch.load('ResNet50-TSX.pth')  
    new_state_dict = checkpoint['model']
    model.load_state_dict({k.replace('module.',''):v for k,v in new_state_dict.items()})
    model.fc = nn.Linear(num_feature, 10)
    model = model.cuda()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    return model


def Model_unet(sample_time):
    model = unet.U_Net(sample_time)
    model = model.cuda()
    return model



def Mag_Train(Model_type, tensorboard_save,model_save_path):
    train_batch = 32
    normalize = transforms.Normalize(mean=[0.051745296], std=[0.059024353])
    train_transform = transforms.Compose([
        
        transform_data.Numpy2Tensor(3),
        transforms.CenterCrop(128),
    ])
    train_list_path = 'downstream tasks/SAR target recognition/data/MSTAR/train.txt'
    save_model_path = model_save_path
    train_dataset = dataset.MsTAR_Dataset(train_list_path, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch,shuffle=True,num_workers=8)
    sample_time = 15
    resnet_model = Model_resnet()
    unet = Model_unet(15)
    optimizer = torch.optim.AdamW([{'params':unet.parameters()},
                                  {'params':filter(lambda p: p.requires_grad, resnet_model.parameters())}], 
                                  eps = 1e-8, lr = 0.00005, weight_decay = 0.05)
    lr_s = lr_scheduler.OneCycleLR(optimizer, max_lr = 0.00001*25,steps_per_epoch=2*(int(5172/(train_batch*1))+1), epochs=250,anneal_strategy='cos')
# start training
    best_acc1 = 0.
    epochs=500
    alpha = 0.25
    loss = []
    writer = SummaryWriter(tensorboard_save)
    for epoch in range(epochs):
        # train for one epoch
        train_loss,entropy,feature_diff,noise = train(train_dataloader, resnet_model,unet, optimizer,lr_s,
             epoch,train_batch,alpha,normalize)
        loss.append(train_loss.item())
        # if train_loss<=np.min(loss):
        if (epoch+1) % 50 == 0:
            # print('Best Epoch:{},Loss:{}'.format(epoch+1,train_loss.item()))
            torch.save(unet.state_dict(), save_model_path+str(epoch+1)+'.pth')
            n = torch.mean(noise,dim=0).cpu().detach().numpy()
            n = ((n-n.min())/(n.max()-n.min()))*255
            n = transform.resize(n, (128, 128))
            cv2.imwrite(save_model_path+str(epoch+1)+'_deltaX.jpg',n)
        if train_loss.item()<=np.min(loss):
            print('Best Epoch:{},Loss:{}'.format(epoch+1,train_loss.item()))
            torch.save(unet.state_dict(), save_model_path+str(epoch)+'best.pth')
            n = torch.mean(noise,dim=0).cpu().detach().numpy()
            n = ((n-n.min())/(n.max()-n.min()))*255
            n = transform.resize(n, (128, 128))
            cv2.imwrite(save_model_path+str(epoch)+'best_deltaX.jpg',n)
        # evaluate on validation set
        # if epoch>=20:
        print(' Epoch: {},Train loss: {}'.format(epoch+1,train_loss.item()))
        print('Feature Diff:{}, Sum of Entropy: {}'.format(feature_diff.item(),entropy.item()))

        writer.add_scalars('loss', {'train': train_loss.item()}, epoch + 1)
        writer.add_scalars('entropy', {'train': entropy.item()}, epoch + 1)
        writer.add_scalars('feature_diff', {'train': feature_diff.item()}, epoch + 1)

def kownledge_point_loss(input_feature,noise_feature,noise,batch,alpha):
    feature_diff = (((noise_feature-input_feature)**2).sum())/batch
    entropy = alpha*torch.sum(noise)
    delta_f = torch.var(input_feature)
    loss = (1/delta_f)*feature_diff-entropy
    return loss,entropy,feature_diff
def train(train_dataloader, resnet,unet, optimizer,scheduler,
             epoch,batch,alpha,normalize):
    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    F = nn.CrossEntropyLoss()
    unet.train()
    resnet.train()
    for idx,(t_data,t_target) in enumerate(train_dataloader):
        true_data,true_target=t_data.cuda(),t_target.cuda()
        input_noise,noise= unet(true_data)
        input_noise_norm = normalize(input_noise)
        true_data_norm = normalize(true_data)
        _,true_data_feature_layer4 = resnet(true_data_norm)
        _,input_noise_feature_layer4 = resnet(input_noise_norm)
        loss,entropy,feature_diff = kownledge_point_loss(true_data_feature_layer4,input_noise_feature_layer4,noise,batch,alpha)
        #Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    # print("epoch:{}, loss:{}".format(epoch,loss.item()))
    return loss,entropy,feature_diff,noise

if __name__ == '__main__':
    Model_type = 'ResNet50'
    UpDataset = 'TSX'
    print('UpDataset:',UpDataset,'----------------------------','Model:',Model_type)
    tensorboard_save = 'knowledge_point_OPT_Pretrain'
    save_path = 'model/optical_pre/'
    Mag_Train(Model_type,tensorboard_save,save_path)