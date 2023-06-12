import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    """ implement conv+ReLU two times """
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)
    def forward(self, x):
        out = self.conv_ReLU(x)
        return out
    
    
    
class Aleatoric_Uncertainty_Quantification1(nn.Module):
    def __init__(self, ori_inputs,channels: int, sample_time:int):
        super().__init__()
        self.channels = channels
        self.sample_time = sample_time
        self.ori_inputs = ori_inputs.cuda()
        self.var = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels, eps=0.001, affine=False),
            ).cuda()
    
    def forward(self, inputs): # B C H W 200*64*16*16
        feature_mu = self.ori_inputs.cuda() # 200*3*128*128
        feature_sqrt_var = torch.exp(self.var(inputs.cuda()) * 0.5).cuda() # 200*64*16*16
        # sqrt_sigma = feature_sqrt_sigma[0,0]
        feature_sqrt_var = torch.mean(feature_sqrt_var,dim=1) #200*16*16
        feature_sqrt_sigma = feature_sqrt_var.unsqueeze(dim = 1) #200*1*16*16
        feature_sqrt_sigma = feature_sqrt_sigma.repeat(1,3,1,1)  # 200*3*16*16
        if self.training:
            sqrt_var = self.up(feature_sqrt_sigma)    #200*3*128*128
            rep_sqrt_var = sqrt_var[None].expand(self.sample_time, *sqrt_var.shape) # S*B*C*H*W S*200*3*128*128
            rep_emb = feature_mu[None].expand(self.sample_time, *feature_mu.shape)# S*200*3*128*128

            norm_v = torch.randn_like(rep_emb,device=rep_emb.device)# S*B*C*H*W
            sto_emb = rep_emb + rep_sqrt_var * norm_v               ### x'
            del norm_v
            return sto_emb,feature_sqrt_var
        else:
            return feature_mu
class U_Net(nn.Module):
    def __init__(self, sample_time:int, ):
        super().__init__()
        self.sample_time = sample_time
        self.up = nn.Upsample(scale_factor = 8 ,mode='nearest')
        self.var = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=0.001, affine=False),
            ).cuda()
        # Left
        self.left_conv_1 = ConvBlock(in_channels=3, middle_channels=64, out_channels=64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = ConvBlock(in_channels=64, middle_channels=128, out_channels=128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = ConvBlock(in_channels=128, middle_channels=256, out_channels=256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.left_conv_4 = ConvBlock(in_channels=256, middle_channels=512, out_channels=512)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_5 = ConvBlock(in_channels=512, middle_channels=1024, out_channels=1024)

        # Right
        self.deconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.right_conv_1 = ConvBlock(in_channels=1024, middle_channels=512, out_channels=512)

    def Aleatoric_Uncertainty_Quantification(self,inputs,ori_inputs,sample_time):
        feature_mu = ori_inputs.cuda() # 200*3*128*128
        feature_sqrt_var = torch.exp(self.var(inputs.cuda()) * 0.5).cuda() # 200*512*16*16

        feature_sqrt_var = torch.mean(feature_sqrt_var,dim=1) #200*16*16
        feature_sqrt_sigma = feature_sqrt_var.unsqueeze(dim = 1) #200*1*16*16
        feature_sqrt_sigma = feature_sqrt_sigma.repeat(1,3,1,1)  # 200*3*16*16

        sqrt_var = self.up(feature_sqrt_sigma)    #200*3*128*128
        rep_sqrt_var = sqrt_var[None].expand(sample_time, *sqrt_var.shape) # S*B*C*H*W S*200*3*128*128
        rep_emb = feature_mu[None].expand(sample_time, *feature_mu.shape)# S*200*3*128*128

        norm_v = torch.randn_like(rep_emb,device=rep_emb.device)# S*B*C*H*W
        sto_emb = rep_emb + rep_sqrt_var * norm_v               ### x'
        del norm_v
        return sto_emb,feature_sqrt_var
    def forward(self, x):

        # Encode
        feature_1 = self.left_conv_1(x) ## [200,3,128,128] -> [200,64,128,128]
        feature_1_pool = self.pool_1(feature_1) ## [200,64,128,128] -> [200,64,64,64]

        feature_2 = self.left_conv_2(feature_1_pool) ## [200,64,64,64] -> [200,128,64,64]
        feature_2_pool = self.pool_2(feature_2)## [200,128,64,64] -> [200,128,32,32]

        feature_3 = self.left_conv_3(feature_2_pool)## [200,128,32,32] -> [200,256,32,32]
        feature_3_pool = self.pool_3(feature_3)## [200,256,32,32] -> [200,256,16,16]
        
        feature_4 = self.left_conv_4(feature_3_pool) ## 256,16,16 -> 512,16,16
        feature_4_pool = self.pool_4(feature_4)     ## 512,16,16 -> 512,8,8

        feature_5 = self.left_conv_5(feature_4_pool) ## 512,8,8 -> 1024,8,8

        # Decode
        de_feature_1 = self.deconv_1(feature_5)  ## 1024,8,8 -> 512,16,16
        # 特征拼接
        temp = torch.cat((feature_4, de_feature_1), dim=1) ## 512,16,16 -> 1024,16,16
        de_feature_1_conv = self.right_conv_1(temp) ## 1024,16,16 -> 512,16,16

        rep_out,sqrt_var = self.Aleatoric_Uncertainty_Quantification(inputs = de_feature_1_conv,ori_inputs=x,sample_time=self.sample_time)
        entropy = torch.log(sqrt_var)+0.5*torch.log(torch.tensor(2*math.pi*math.e))
        out = torch.mean(rep_out,dim=0)
        return out,entropy