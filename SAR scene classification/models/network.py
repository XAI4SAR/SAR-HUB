from turtle import forward
from matplotlib.pyplot import xcorr
import torch.nn as nn
import torch.nn.init as init
import torch
import math
from collections import OrderedDict
from typing import Any, Tuple
import torch.nn.functional as F
from torch import Tensor
class AlexNet_TSX(nn.Module):
    def __init__(self, num_class=7):
        super(AlexNet_TSX, self).__init__()
        # input image size: 160```
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, (11, 11), (4, 4)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2), 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Conv2d(384, 384, (3, 3), (1, 1), (1, 1), 1, 2),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1), 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
        )
        self.classifier = nn.Sequential(
            # nn.Linear(256*3*3, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(256, 256),
            # nn.LeakyReLU(),
            # nn.Dropout(0.7),
            # nn.Linear(128, num_class)
            nn.Linear(256*3*3, num_class) # 256*3*3 is for input of 128*128, to be modified
        )
        self._initialize_weights()
        self.layer_dict = {'conv1': 'features.0', 'conv2': 'features.4', 'conv3': 'features.8', 'conv4': 'features.11',
                           'conv5': 'features.14', 'maxpool5': 'features.17', 'fc1': 'classifier.0'
            # , 'fc2': 'classifier.2'#, 'fc3': 'classifier.4'
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        """
            use He's initializer
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class AlexNet_OpenSAR(nn.Module):
    def __init__(self, num_class=3):
        super(AlexNet_OpenSAR, self).__init__()
        # input image size: 160```
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, (11, 11), (4, 4)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2), 1, 2),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 384, (3, 3), (1, 1), (1, 1), 1, 2),
            nn.ReLU(),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1), 1, 2),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*2*2, num_class)
        )
        self._initialize_weights()
        self.layer_dict = {'conv1': 'features.0', 'conv2': 'features.3', 'conv3': 'features.6', 'conv4': 'features.8',
                           'conv5': 'features.10', 'maxpool5': 'features.12', 'fc1': 'classifier.0'}

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        """
            use He's initializer
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

""" ResNet """
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.5)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#* 定义Flatten层，降维，多维压至一维
class Flatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0),-1)
#* DOWN!
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=7):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1) # for input of 128*128
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # self.fc1 = nn.Linear(512 * block.expansion * 7 * 7, 512)
        # self.bn2 = nn.BatchNorm1d(512 * block.expansion, eps=2e-5, affine=False)
        # self.drop = nn.Dropout(p=0.5)
        # self.bn3 = nn.BatchNorm1d(512, eps=2e-5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    #! Data uncertainty in face recognition
        # #* μ (features/mean)
        # self.mu_head = nn.Sequential(
        #     Flatten(),
        #     nn.BatchNorm1d(512 * block.expansion, eps=2e-5, affine=False),
        #     nn.Dropout(p=0.1),
        #     # Flatten(),
        #     nn.Linear(512 * block.expansion, 512),
        #     nn.BatchNorm1d(512, eps=2e-5))
        # #* log var (uncertainty/var)
        # self.logvar_head = nn.Sequential(
        #     Flatten(),
        #     nn.BatchNorm1d(512 * block.expansion, eps=2e-5, affine=False),
        #     nn.Dropout(p=0.1),
        #     # Flatten(),
        #     nn.Linear(512 * block.expansion, 512),
        #     nn.BatchNorm1d(512, eps=2e-5))
    #! Data uncertainty DOWN!
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.5),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
#! 换权重，w换成s，s=mu+epsilon*std
    def _reparameterize(self,mu,logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu+epsilon*std
#! DOWN！
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
    #* 换权重X为新权重s
        # mu = self.mu_head(x)
        # logvar = self.logvar_head(x)
        # x = self._reparameterize(mu, logvar)
    #* DOWN!
        # x = x.view(x.size(0), -1)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.fc(x) #? outputs

        # return x, torch.sum(torch.abs(feats), 1).reshape(-1, 1), mu, logvar
        return x
class ResNet_uncertainty(nn.Module):

    def __init__(self, block, layers, num_classes=7):
        self.inplanes = 64
        super(ResNet_uncertainty, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.5)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1) # for input of 128*128
        self.cls_layer = nn.Linear(512 * block.expansion, num_classes)
        # self.fc1 = nn.Linear(512 * block.expansion * 7 * 7, 512)
        # self.bn2 = nn.BatchNorm1d(512 * block.expansion, eps=2e-5, affine=False)
        # self.drop = nn.Dropout(p=0.5)
        # self.bn3 = nn.BatchNorm1d(512, eps=2e-5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

        ################! uncertainty estimation ##################### 
        self.to_vector = nn.Sequential(
            nn.Linear(512*8*8,512),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.emd = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
        )
        self.var = nn.Sequential(
            nn.Linear(512,512),
            nn.BatchNorm1d(512, eps=0.001, affine=False),
        )
        self.final = nn.Linear(512,num_classes)
        self.drop = nn.Dropout()
        ################! uncertainty estimation #####################
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.5),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def uncertainty_quantification(self,vector_sigma): #100*512
        n = vector_sigma.shape[-1]
        vector_sigma = torch.reciprocal(torch.exp(vector_sigma))
        data_uncertainty = n / torch.sum(vector_sigma,dim=1,keepdim=True)
        return data_uncertainty
    def forward(self, x, training = True):
        ###? backbone ####
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # 512*256*8*8
        x = self.layer4(x) # 512*512*4*4
        ###? backbone down ###
        ###! uncertainty estimation ###
        vector = self.avgpool(x) # 512*512*1*1
        vector = vector.view(vector.size(0), -1) # 512*512
        vector_mu = self.emd(vector)    #* mean/feature 512*512
        #! 此处的vector_var实际上是logvar
        vector_var = self.var(vector)   #* var/uncertainty 512*512 
        if training:
            vector_sqrt_var = 0.5*torch.exp(vector_var)
            rep_emb = vector_mu[None].expand(50, *vector_mu.shape) # 50*512*512 #* expand 复制50个
            rep_sqrt_var = vector_sqrt_var[None].expand(50, *vector_var.shape) # 50*512*512 #* expand 复制50个
            norm_v = torch.randn_like(rep_emb).cuda() # 50*512*512
            sto_emb = rep_emb + rep_sqrt_var * norm_v # 50*512*512
            # sto_emb = self.drop(sto_emb) # 50*512*512
            logit = self.final(sto_emb) # 50*512*#!45
        else:
            logit = self.final(vector_mu)
            rep_emb = 0
           
        # x = x.view(x.size(0), -1)
        # x = self.avgpool(x)
        # x = self.avgpool(x)
        # feats = x.view(x.size(0), -1)
        # x = self.cls_layer(feats) #? outputs
        
        # return x, torch.sum(torch.abs(feats), 1).reshape(-1, 1), mu, logvar
        data_uncertainty = self.uncertainty_quantification(vector_var)
        
        return logit, data_uncertainty, rep_emb,vector_var
    
    
#! MobilenetV3-Small

def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=45, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
#! Mobilenetv3-Small
def ResNet18():
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=1000)

    return model

def ResNet_TSX(tsx_num_class):
    model = ResNet(BasicBlock, [1,1,1,1], num_classes=tsx_num_class)

    return model

def ResNet18_TSX(tsx_num_class):
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=tsx_num_class)
    return model

def ResNet18_opti_rs(tsx_num_class):
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=tsx_num_class)
    return model

def ResNet18_TSX_Comp_CNN2_data_uncertainty_for_CNN_feature(tsx_num_class):
    model = ResNet_uncertainty(BasicBlock, [2,2,2,2], num_classes=tsx_num_class)
    return model
def ResNet50(tsx_num_classes):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=tsx_num_classes)
    return model
def ResNet50_opt_aid(tsx_num_classes):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=tsx_num_classes)
    return model
def ResNet50_opt_nwpu(tsx_num_classes):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=tsx_num_classes)
    return model
def ResNet101(tsx_num_classes):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=tsx_num_classes)
    return model
def get_features_by_name(model, x, layer_name):
    layer_dict = model.layer_dict
    layer_num = layer_dict[layer_name]
    for i, l in enumerate(model.features):
        x = l(x)
        if 'features.' + str(i) == layer_num:
            break

    return x.data[0]


#! densenet
# 定义densenet的最基本模块，包含BN1 + relu1 + conv1 + BN2 + relu2 + conv2 + dropout，注意这里是BN在最前面，一般别的模型都是conv在前
class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1: nn.BatchNorm2d                                    # 定义norm1这个字段并提前赋予数据类型
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))  # 对定义的norm1字段进行赋值
        self.relu1: nn.ReLU
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module(                                              # 第一个卷积模块输出通道数是bn_size * growth_rate
            "conv1", nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        )
        self.norm2: nn.BatchNorm2d
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module(                                              # 第二个卷积模块输出通道数是growth_rate
            "conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.drop_rate = float(drop_rate)
 
    def forward(self, input: Tensor) -> Tensor:
        prev_features = input
        new_features0 = torch.cat(prev_features, 1)                   # 每一个最基本模块的输入通道是init_num + (n - 1) * growth_rate
        new_features1 = self.conv1(self.relu1(self.norm1(new_features0)))  # 第一个卷积输出通道bn_size * growth_rate
        new_features2 = self.conv2(self.relu2(self.norm2(new_features1)))  # 第二个卷积输出通道growth_rate
        if self.drop_rate > 0:                                        # 每一个最基本模块的输出通道是growth_rate
            new_features2 = F.dropout(new_features2, p=self.drop_rate, training=self.training)  # 当前使用时，没有启用这一层
        return new_features2
 
# 定义densenet的大模块，包含num_layers个最基本模块，这个num_layers个最基本模块遵循密集连接的原则
# nn.ModuleDict可以以字典的形式向nn.ModuleDict中输入子模块，也可以以add_module()的形式向nn.ModuleDict中输入子模块
# 但是nn.ModuleDict类似于nn.Module需要自己实现forward()函数，类似的模块还有nn.ModuleList以列表形式搭建模型
# 所以说白了nn.Sequential，nn.Module，nn.ModuleList，nn.ModuleDict是搭建模型或模块的四种方式，是并行的关系，可以根据不同应用条件下使用
# https://blog.csdn.net/weixin_42486623/article/details/122822580
class _DenseBlock(nn.ModuleDict):
 
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate
            )
            self.add_module("denselayer%d" % (i + 1), layer)   # 以add_module()形式输入子模块
 
    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():                       # 以items()形式访问子模块
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
 
# 定义densenet的大模块，用于拼接_DenseBlock模块，在本模块内通过均值池化将空间尺寸减小一半
# torch.nn.Sequential相当于tf2.0中的keras.Sequential()，其实就是以最简单的方式搭建序列模型，不需要写forward()函数，
# 直接以列表形式将每个子模块送进来就可以了，或者也可以使用OrderedDict()或add_module()的形式向模块中添加子模块
# https://blog.csdn.net/weixin_42486623/article/details/122822580
class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))
 
# 根据block_config参数配置列表搭建整个densenet模型
class DenseNet(nn.Module):
 
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
    ) -> None:
 
        super().__init__()
 
        ### 搭建第一层，即stem层，包含conv + BN + relu + maxpool，以字典的形式向nn.Sequential中添加子模块
        self.features = nn.Sequential(        # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
 
        ### 搭建bottleneck层，包含4个_DenseBlock大模块和4个_Transition大模块
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2  # _Transition模块不仅将空间尺寸减半还将通道尺寸减半
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
 
        ### 搭建最后的分类层
        self.classifier = nn.Linear(num_features, num_classes)
 
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
 
##############################################################################################################################
## 通过修改配置列表实现不同模型的定义
def densenet121(target_num_classes,**kwargs: Any) -> DenseNet:
    return DenseNet(32, (6, 12, 24, 16), 64,num_classes=target_num_classes, **kwargs)
 
def densenet161(**kwargs: Any) -> DenseNet:
    return DenseNet(48, (6, 12, 36, 24), 96, **kwargs)
 
def densenet169(**kwargs: Any) -> DenseNet:
    return DenseNet(32, (6, 12, 32, 32), 64, **kwargs)
 
def densenet201(**kwargs: Any) -> DenseNet:
    return DenseNet(32, (6, 12, 48, 32), 64, **kwargs)
#!

#! Senet50
class Block(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_1x1conv=False):
        super(Block, self).__init__()
        filter1, filter2, filter3 = filters
        self.is_1x1conv = is_1x1conv
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride,bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filter3),
        )
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filter3)
            )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(filter3,filter3//16,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(filter3//16,filter3,kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x_shortcut = x
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.se(x1)
        x1 = x1*x2
        if self.is_1x1conv:
            x_shortcut = self.shortcut(x_shortcut)
        x1 = x1 + x_shortcut
        x1 = self.relu(x1)
        return x1

class senet(nn.Module):
    def __init__(self,cfg):
        super(senet,self).__init__()
        classes = cfg['classes']
        num = cfg['num']
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(64, (64, 64, 256), num[0],1)
        self.conv3 = self._make_layer(256, (128, 128, 512), num[1], 2)
        self.conv4 = self._make_layer(512, (256, 256, 1024), num[2], 2)
        self.conv5 = self._make_layer(1024, (512, 512, 2048), num[3], 2)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048,classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self,in_channels, filters, num, stride=1):
        layers = []
        block_1 = Block(in_channels, filters, stride=stride, is_1x1conv=True)
        layers.append(block_1)
        for i in range(1, num):
            layers.append(Block(filters[2], filters, stride=1, is_1x1conv=False))
        return nn.Sequential(*layers)

def Senet(block,num_classes):
    cfg = {
        'num':block,
        'classes': num_classes
    }
    return senet(cfg)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=45, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs,mode='small', **kwargs)