import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import net.resnet as res

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Resnet50_NL(nn.Module):
    def __init__(self,non_layers=[0,1,1,1],stripes=[16,16,16,16],non_type='normal',temporal=None):
        super(Resnet50_NL,self).__init__()
        original = models.resnet50(pretrained=True).state_dict()
        if non_type == 'normal':
            self.backbone = res.ResNet_Video_nonlocal(last_stride=1,non_layers=non_layers)
        elif non_type == 'stripe':
            self.backbone = res.ResNet_Video_nonlocal_stripe(last_stride = 1, non_layers=non_layers, stripes=stripes)
        elif non_type == 'hr':
            self.backbone = res.ResNet_Video_nonlocal_hr(last_stride = 1, non_layers=non_layers, stripes=stripes)
        elif non_type == 'stripe_hr':
            self.backbone = res.ResNet_Video_nonlocal_stripe_hr(last_stride = 1, non_layers=non_layers, stripes=stripes)
        for key in original:
            if key.find('fc') != -1:
                continue
            self.backbone.state_dict()[key].copy_(original[key])
        del original

        self.temporal = temporal
        if self.temporal == 'Done':
            self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self,x):
        if self.temporal == 'Done':
            x = self.backbone(x)
            x = self.avgpool(x)
            x = x.reshape(x.shape[0],-1)
            return x


class Resnet50_s1(nn.Module):
    def __init__(self,pooling=True,stride=1):
        super(Resnet50_s1,self).__init__()
        original = models.resnet50(pretrained=True).state_dict()
        self.backbone = res.ResNet(last_stride=stride)
        for key in original:
            if key.find('fc') != -1:
                continue
            self.backbone.state_dict()[key].copy_(original[key])
        del original
        if pooling == True:
            self.add_module('avgpool',nn.AdaptiveAvgPool2d(1))
        else:
            self.avgpool = None

        self.out_dim = 2048

    def forward(self,x):
        x = self.backbone(x)
        if self.avgpool is not None:
            x = self.avgpool(x)
            x = x.view(x.shape[0],-1)
        return x

class CNN(nn.Module):
    def __init__(self,out_dim,model_type='resnet50_s1',num_class=710,non_layers=[1,2,2],stripes=[16,16,16,16], temporal = 'Done',stride=1):
        super(CNN,self).__init__()
        self.model_type = model_type
        if model_type == 'resnet50_s1':
            self.features = Resnet50_s1(stride=stride)
        elif model_type == 'resnet50_NL':
            self.features = Resnet50_NL(non_layers=non_layers,temporal=temporal,non_type='normal')
        elif model_type == 'resnet50_NL_stripe':
            self.features = Resnet50_NL(non_layers=non_layers,stripes=stripes,temporal=temporal,non_type='stripe')
        elif model_type == 'resnet50_NL_hr':
            self.features = Resnet50_NL(non_layers=non_layers,stripes=stripes,temporal=temporal,non_type='hr')
        elif model_type == 'resnet50_NL_stripe_hr':
            self.features = Resnet50_NL(non_layers=non_layers,stripes=stripes,temporal=temporal,non_type='stripe_hr')

        self.bottleneck = nn.BatchNorm1d(out_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(out_dim,num_class, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self,x,seg=None):
        if self.model_type == 'resnet50_s1':
            x = self.features(x)
            bn = self.bottleneck(x)
            if self.training == True:
                output = self.classifier(bn)
                return x,output
            else:
                return bn
        elif self.model_type == 'resnet50_NL' or self.model_type == 'resnet50_NL_stripe' or \
            self.model_type=='resnet50_NL_hr' or self.model_type == 'resnet50_NL_stripe_hr':
            x = self.features(x)
            bn = self.bottleneck(x)
            if self.training == True:
                output = self.classifier(bn)
                return x,output
            else:
                return bn

if __name__ == '__main__':
    model = Resnet50_s1()
    input = torch.ones(1,3,256,128)
    output = model(input)
    print(output.shape)
