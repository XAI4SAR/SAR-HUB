import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(ce_loss, gamma,alpha):
    """Computes the focal loss"""
    p = torch.exp(-ce_loss)
    #loss = (1 - p) ** gamma * input_values
    loss = (alpha*(1- p) ** gamma * ce_loss).mean() 
    return loss.mean()

def CB_Focal(ce_loss, gamma, alpha):
    p = torch.exp(-ce_loss)
    #loss = (1 - p) ** gamma * input_values
    loss = (alpha*(1- p) ** gamma * ce_loss).mean() 
    return loss.mean()
class MINI_CB_Focal_Loss(nn.Module):
    def __init__(self,samples_per_cls, beta, num_of_cls, gamma):
        super(MINI_CB_Focal_Loss,self).__init__()
        self.samples_per_cls = samples_per_cls
        self.beta = beta
        self.num_of_cls = num_of_cls
        self.gamma = gamma

    def forward(self, input ,labels):
        label = labels.cpu()
        label = (label.numpy().tolist())
        
        sample_per_batch = []
        for i in range(len(self.num_of_cls)):
            sample_per_batch.append(label.count(i))
        sample_per_batch = torch.tensor(sample_per_batch)
        sample_per_batch = torch.clamp(sample_per_batch,1)
        effective_numbers = 1.0-np.power(self.beta, sample_per_batch)
        weights = (1.0 - self.beta) / np.array(effective_numbers)
        weights = torch.tensor(weights).float().cuda()
        cross_loss = F.cross_entropy(input,labels,weights)
        cb_loss = CB_Focal(cross_loss ,self.gamma, weights)
        return cb_loss
    
class CB_Focal_Loss(nn.Module):
    def __init__(self,samples_per_cls, beta, num_of_cls, gamma):
        super(CB_Focal_Loss,self).__init__()
        self.samples_per_cls = samples_per_cls
        self.beta = beta
        self.num_of_cls = num_of_cls
        self.gamma = gamma

    def forward(self, input ,labels):
        samples_cls = self.samples_per_cls.cpu()
        effective_numbers = 1.0-np.power(self.beta, samples_cls)
        weights = (1.0 - self.beta) / np.array(effective_numbers)
        weights = torch.tensor(weights).float().cuda()
        cross_loss = F.cross_entropy(input,labels,weight=weights).cuda()
        cb_loss = CB_Focal(cross_loss ,self.gamma, weights)
        return cb_loss