import math
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import os
class ReID_NonLocalBlock(nn.Module):
    def __init__(self,stripe,in_channels,inter_channels=None,type='mean'):
        super(ReID_NonLocalBlock,self).__init__()
        self.stripe=stripe
        self.in_channels = in_channels
        self.type = type
        if type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif type == 'mean':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif type == 'meanmax':
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.maxpool = nn.AdaptiveMaxPool2d(1)
            self.in_channels*=2
        if inter_channels == None:
            self.inter_channels = in_channels//2 
        else:
            self.inter_channels = inter_channels
        self.g = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        if self.type == 'meanmax':
            self.in_channels //=2
        self.W = nn.Sequential(
            nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
    def forward(self,x):
        # x.shape = (b,c,t,h,w)
        b,c,t,h,w = x.shape
        assert self.stripe * (h//self.stripe) == h
        if self.type == 'meanmax': 
            discri_a = self.avgpool(x.reshape(b*c*t,self.stripe,(h//self.stripe),w)).reshape(b,c,t,self.stripe,1)
            discri_m = self.maxpool(x.reshape(b*c*t,self.stripe,(h//self.stripe),w)).reshape(b,c,t,self.stripe,1)
            discri = torch.cat([discri_a,discri_m],dim=1)
        else:
            discri = self.pool(x.reshape(b*c*t,self.stripe,(h//self.stripe),w)).reshape(b,c,t,self.stripe,1)
        g = self.g(discri).reshape(b,self.inter_channels,-1)
        g = g.permute(0,2,1)
        theta = self.theta(discri).reshape(b, self.inter_channels, -1)
        theta = theta.permute(0,2,1)
        phi = self.phi(discri).reshape(b, self.inter_channels, -1)

        f = torch.matmul(theta, phi)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g)
        y = y.permute(0, 2, 1).contiguous()
        y = y.reshape(b, self.inter_channels, *discri.size()[2:])
        W_y = self.W(y)

        W_y = W_y.repeat(1,1,1,1,h//self.stripe*w).reshape(b,c,t,h,w)

        z = W_y + x
        return z 

class cross_NonLocalBlock(nn.Module):
    def __init__(self,in_channels,inter_channels=None,bn_layer=True,instance='soft'):
        super(cross_NonLocalBlock,self).__init__()
        self.instance = instance
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.g = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.g_low = nn.Conv3d(in_channels=self.inter_channels,out_channels=self.inter_channels,
                        kernel_size = 1,stride=1,padding=0)
        
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.phi_low = nn.Conv3d(in_channels=self.inter_channels,out_channels=self.inter_channels,
                                 kernel_size=1,stride=1,padding=0)
    def forward(self, x , x_low):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert x.shape[1] == (x_low.shape[1]*2)
        batch_size = x.size(0)

        g_x = self.g(x).reshape(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        g_x_low = self.g_low(x_low).reshape(batch_size,self.inter_channels,-1)
        g_x_low = g_x_low.permute(0,2,1)
        cat_g = torch.cat([g_x,g_x_low],dim=1)

        theta_x = self.theta(x).reshape(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).reshape(batch_size, self.inter_channels, -1)
        phi_x_low = self.phi_low(x_low).reshape(batch_size,self.inter_channels,-1)
        cat_phi = torch.cat([phi_x,phi_x_low],dim=2)

        f = torch.matmul(theta_x, cat_phi)
        if self.instance == 'soft':
            f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, cat_g)
        y = y.permute(0, 2, 1).contiguous()
        y = y.reshape(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        
        return z


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=False, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
        
    def forward(self, input):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        if type(input) == type((1,)):
            x = input[0]
        else : x = input

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        # f_div_C = f / f.shape[1]


        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        
        if type(input) == type((1,)):
            z = (x,f_div_C)
        return z

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

class ResNet_ReID_nonlocal(nn.Module):
    def __init__(self,last_stride=2,block=Bottleneck,layers=[3,4,6,3],non_layers=[1,1,1]):
        self.inplanes = 64
        super().__init__()
        type = 'mean'
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        #self.NL_1 = nn.Sequential(*[ReID_NonLocalBlock(64//2,self.inplanes,type=type) for i in range(non_layers[0])])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.NL_2 = nn.Sequential(*[ReID_NonLocalBlock(16,self.inplanes,type=type) for i in range(non_layers[0])])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.NL_3 = nn.Sequential(*[ReID_NonLocalBlock(16,self.inplanes,type=type) for i in range(non_layers[1])])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.NL_4 = nn.Sequential(*[ReID_NonLocalBlock(16,self.inplanes,type=type) for i in range(non_layers[2])])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x (B,T,C,H,W)
        B,T,C,H,W = x.shape
        x = x.reshape(B*T,C,H,W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_2(x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)

        x = self.layer3(x)
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x  = self.NL_3(x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        
        x = self.layer4(x)
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_4(x)
        # x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)

        return x

class ResNet_nonlocal(nn.Module):
    def __init__(self,last_stride=2,block=Bottleneck,layers=[3,4,6,3],non_layers=[1,2,2],soft_back = False):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.NL_2 = nn.Sequential(*[_NonLocalBlockND(self.inplanes) for i in range(non_layers[0])])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.NL_3 = nn.Sequential(*[_NonLocalBlockND(self.inplanes) for i in range(non_layers[1])])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.NL_4 = nn.Sequential(*[_NonLocalBlockND(self.inplanes) for i in range(non_layers[2])])
        self.soft_back = soft_back

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x (B,T,C,H,W)
        B,T,C,H,W = x.shape
        x = x.reshape(B*T,C,H,W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x = self.NL_2(x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)

        x = self.layer3(x)
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        x  = self.NL_3(x)
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        
        x = self.layer4(x)
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        if self.soft_back == True:
            x  = self.NL_4((x,self.soft_back))
        else:
            x = self.NL_4(x)
        # x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)

        return x

class ResNet_cross_nonlocal(nn.Module):
    def __init__(self,last_stride=1,block=Bottleneck,layers=[3,4,6,3],non_layers=[1,2,2]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.NL_2 = nn.ModuleList( [_NonLocalBlockND(self.inplanes,self.inplanes//2) for i in range(non_layers[0])])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.NL_3 = nn.ModuleList( [ cross_NonLocalBlock(self.inplanes,self.inplanes//2) for i in range(non_layers[1])])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.NL_4 = nn.ModuleList( [ cross_NonLocalBlock(self.inplanes,self.inplanes//2) for i in range(non_layers[1])])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x (B,T,C,H,W)
        B,T,C,H,W = x.shape
        x = x.reshape(B*T,C,H,W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        # ------------------
        x = self.layer1(x)
        # ------------------
        x = self.layer2(x)
        # -----Non-local----
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        for i in range(len(self.NL_2)):
            x = self.NL_2[i](x)
        L2 = x
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        # ------------------
        x = self.layer3(x)
        # -----Non-local----
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        for i in range(len(self.NL_3)):
            x  = self.NL_3[i](x,L2)
        L3 = x
        x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
        # ------------------
        x = self.layer4(x)
        # -----Non-local----
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        for i in range(len(self.NL_4)):
            x = self.NL_4[i](x,L3)
        # x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)

        return x
class SpatialGate(nn.Module):
    def __init__(self,kernel_size=5,conv_dim=2):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        if conv_dim == 2:
            self.conv = nn.Conv2d(2,1,kernel_size,padding=(kernel_size-1)//2,bias=False)
            self.norm = nn.BatchNorm2d(1,eps=1e-5,momentum=0.01,affine=True)
        elif conv_dim == 3:
            self.conv = nn.Conv3d(2,1,kernel_size,padding=(kernel_size-1)//2,bias=False)
            self.norm = nn.BatchNorm3d(1, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        self.conv_dim = conv_dim
    def forward(self, x):
        x_compress = torch.cat( (torch.max(x,1,keepdim=True)[0] , torch.mean(x,1,keepdim=True)), dim=1 )
        x_out = self.norm(self.conv(x_compress))
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class ResNet_ReID_realnonlocal(nn.Module):
    def __init__(self,last_stride=2,block=Bottleneck,layers=[3,4,6,3],non_layers=[1,1,1]):
        self.inplanes = 64
        super().__init__()
        type = 'mean'
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.NL_2 = nn.ModuleList([ReID_NonLocalBlock(16,self.inplanes,type=type) for i in range(non_layers[0])])
        self.NL_2_idx = sorted([layers[1]-(i+1) for i in range(non_layers[0])])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.NL_3 = nn.ModuleList([ReID_NonLocalBlock(16,self.inplanes,type=type) for i in range(non_layers[1])])
        self.NL_3_idx =sorted( [layers[2]-(i+1) for i in range(non_layers[1])])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.NL_4 = nn.ModuleList([ReID_NonLocalBlock(16,self.inplanes,type=type) for i in range(non_layers[2])])
        self.NL_4_idx = sorted([layers[3]-(i+1) for i in range(non_layers[2])])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.ModuleList(layers)

    def forward(self, x):
        # x (B,T,C,H,W)
        B,T,C,H,W = x.shape
        x = x.reshape(B*T,C,H,W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        NL2_counter = 0
        if len(self.NL_2_idx)==0: self.NL_2_idx=[len(self.layer2)]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _,C,H,W = x.shape
                x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
                x = self.NL_2[NL2_counter](x)
                x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
                NL2_counter+=1
        NL3_counter = 0
        if len(self.NL_3_idx)==0: self.NL_3_idx=[len(self.layer3)]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _,C,H,W = x.shape
                x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
                x = self.NL_3[NL3_counter](x)
                x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
                NL3_counter+=1
        NL4_counter = 0
        if len(self.NL_4_idx)==0: self.NL_4_idx=[len(self.layer4)]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _,C,H,W = x.shape
                x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
                x = self.NL_4[NL4_counter](x)
                x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
                NL4_counter+=1
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        return x
class ResNet_realnonlocal(nn.Module):
    def __init__(self,last_stride=2,block=Bottleneck,layers=[3,4,6,3],non_layers=[1,1,1]):
        self.inplanes = 64
        super().__init__()
        type = 'mean'
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.NL_2 = nn.ModuleList([_NonLocalBlockND(self.inplanes,self.inplanes//2) for i in range(non_layers[0])])
        self.NL_2_idx = sorted([layers[1]-(i+1) for i in range(non_layers[0])])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.NL_3 = nn.ModuleList([_NonLocalBlockND(self.inplanes,self.inplanes//2)  for i in range(non_layers[1])])
        self.NL_3_idx =sorted( [layers[2]-(i+1) for i in range(non_layers[1])])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.NL_4 = nn.ModuleList([_NonLocalBlockND(self.inplanes,self.inplanes//2) for i in range(non_layers[2])])
        self.NL_4_idx = sorted([layers[3]-(i+1) for i in range(non_layers[2])])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.ModuleList(layers)

    def forward(self, x):
        # x (B,T,C,H,W)
        B,T,C,H,W = x.shape
        x = x.reshape(B*T,C,H,W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)

        NL2_counter = 0
        if len(self.NL_2_idx)==0: self.NL_2_idx=[len(self.layer2)]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _,C,H,W = x.shape
                x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
                x = self.NL_2[NL2_counter](x)
                x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
                NL2_counter+=1
        NL3_counter = 0
        if len(self.NL_3_idx)==0: self.NL_3_idx=[len(self.layer3)]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _,C,H,W = x.shape
                x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
                x = self.NL_3[NL3_counter](x)
                x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
                NL3_counter+=1
        NL4_counter = 0
        if len(self.NL_4_idx)==0: self.NL_4_idx=[len(self.layer4)]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _,C,H,W = x.shape
                x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
                x = self.NL_4[NL4_counter](x)
                x = x.permute(0,2,1,3,4).reshape(B*T,C,H,W)
                NL4_counter+=1
        _,C,H,W = x.shape
        x = x.reshape(B,T,C,H,W).permute(0,2,1,3,4)
        return x
if __name__ == "__main__":

    # non = cross_NonLocalBlock(in_channels=1024,inter_channels=512)
    # non = ReID_NonLocalBlock(8,1024,512)


    # x = torch.ones(2,1024,6,16,8)

    # output = non(x)
    # exit(-1) 

    r = ResNet_ReID_nonlocal(last_stride=1)
    input = torch.ones(2,6,3,256,128)
    r(input)
