import torch.nn as nn
import torch
import torch.nn.init
from torch.autograd import Variable
import numpy as np

INIT_CONST = 1.18

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity=nn.ReLU, stride=1):
        super(ResLayer, self).__init__()
        self.stride = stride
        self.pad = out_channels - in_channels

        self.layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nonlinearity(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            ]
        self.features = nn.Sequential(*self.layers)

        if stride > 1:
            self.downsample = nn.AvgPool2d(1, stride=stride)

    def forward(self, x):
        skip = x
        if self.stride > 1:
            skip = self.downsample(skip)
        if self.pad > 0:
            sz = list(skip.size())
            sz[1] = self.pad
            zeros = Variable(torch.zeros(*sz).cuda(), requires_grad=False)
            skip = torch.cat((skip, zeros), 1)

        feats = self.features(x)
        return skip + feats

    def kaiming_init(self):
        for l in self.layers:
            name = type(l).__name__
            if name == 'Conv2d':
                torch.nn.init.kaiming_normal(l.weight)
                l.bias.data.zero_()
            elif name == 'BatchNorm2d':
                l.weight.data.normal_(1.0, 0.002)
                l.bias.data.zero_()

    def greg_init(self, depth):
        num_conv = 0
        scale = INIT_CONST ** (-depth)
        for l in self.layers:
            name = type(l).__name__
            if name == 'Conv2d':
                torch.nn.init.kaiming_normal(l.weight)
                l.bias.data.zero_()
                l.weight.data.mul_(scale)
                num_conv += 1
            elif name == 'BatchNorm2d':
                l.weight.data.normal_(1.0, 0.002)
                l.bias.data.zero_()

    def register_hook(self, hook):
        for l in self.layers:
            name = type(l).__name__
            if name == 'ReLU':
                l.register_backward_hook(hook)

class ResConvNet(nn.Module):
    def __init__(self, num_layers, nonlinearity=nn.ReLU):
        super(ResConvNet, self).__init__()
        self.num_layers = num_layers

        self.layers = [
            nn.Conv2d(3, 16, 3, stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nonlinearity(),
            ]

        for _ in range(num_layers):
            self.layers.append(ResLayer(16, 16, nonlinearity=nonlinearity))

        self.layers.append(ResLayer(16, 32, nonlinearity=nonlinearity, stride=2))
        for _ in range(num_layers - 1):
            self.layers.append(ResLayer(32, 32, nonlinearity=nonlinearity))

        self.layers.append(ResLayer(32, 64, nonlinearity=nonlinearity, stride=2))
        for _ in range(num_layers - 1):
            self.layers.append(ResLayer(64, 64, nonlinearity=nonlinearity))

        self.layers.append(nn.AvgPool2d(8))

        self.features = nn.Sequential(*self.layers)
        self.classifier = nn.Linear(64, 10)
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(-1, 64))
        return output

    def kaiming_init(self):
        for l in self.layers:
            name = type(l).__name__
            if name == 'ResLayer':
                l.kaiming_init()
            elif name == 'Conv2d':
                torch.nn.init.kaiming_normal(l.weight)
                l.bias.data.zero_()

    def greg_init(self):
        # Currently reversed
        num_res_layer = 0
        tot_layers = self.num_layers * 3 - 1
        for l in self.layers:
            name = type(l).__name__
            if name == 'ResLayer':
                l.greg_init(tot_layers - num_res_layer)
                num_res_layer += 1
            elif name == 'Conv2d':
                # same as before, 1st layer
                torch.nn.init.kaiming_normal(l.weight)
                l.bias.data.zero_()

        # last layer
        # scale = INIT_CONST ** (- num_res_layer * self.num_layers)
        # self.classifier.weight.data.mul_(scale)
        # self.classifier.bias.data.mul_(scale)

    def register_hook(self, save_grad):
        num_res_layer = 0
        for l in self.layers:
            name = type(l).__name__
            if name == 'ReLU':
                l.register_backward_hook(save_grad('first'))
            elif name == 'ResLayer':
                layer_name = 'reslayer_{}'.format(num_res_layer)
                l.register_hook(save_grad(layer_name))
                num_res_layer += 1


