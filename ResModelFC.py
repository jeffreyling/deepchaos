import torch.nn as nn
import torch
import torch.nn.init
from torch.autograd import Variable
import numpy as np
import pdb

INIT_CONST = 1.18

class ResLayerFC(nn.Module):
    def __init__(self, num_hid, nonlinearity=nn.ReLU):
        super(ResLayerFC, self).__init__()

        self.layers = [
            nn.Linear(num_hid, num_hid),
            # nn.BatchNorm1d(num_hid),
            nonlinearity(),
            ]
        self.features = nn.Sequential(*self.layers)

    def forward(self, x):
        output = x + self.features(x)
        return output

    def kaiming_init(self):
        for l in self.layers:
            name = type(l).__name__
            if name == 'Linear':
                torch.nn.init.kaiming_normal(l.weight)
                l.bias.data.zero_()
            elif name == 'BatchNorm1d':
                l.weight.data.normal_(1.0, 0.002)
                # l.weight.data.normal_(0.0, 0.002)
                l.bias.data.zero_()

    def greg_init(self, depth):
        num_conv = 0
        scale = INIT_CONST ** (-depth)
        for l in self.layers:
            name = type(l).__name__
            if name == 'Linear':
                torch.nn.init.kaiming_normal(l.weight)
                l.bias.data.zero_()
                l.weight.data.mul_(scale)
                num_conv += 1
            elif name == 'BatchNorm1d':
                l.weight.data.normal_(1.0, 0.002)
                l.bias.data.zero_()

    def register_hook(self, hook):
        for l in self.layers:
            name = type(l).__name__
            if name == 'ReLU' or name == 'Tanh':
                l.register_backward_hook(hook)

class ResNetFC(nn.Module):
    def __init__(self, num_hid, num_layers, nonlinearity=nn.ReLU):
        super(ResNetFC, self).__init__()
        self.num_layers = num_layers

        self.layers = [
            nn.Linear(3*32*32, num_hid),
            # nn.BatchNorm1d(num_hid),
            nonlinearity(),
            ]

        for _ in range(num_layers):
            self.layers.append(ResLayerFC(num_hid, nonlinearity=nonlinearity))

        self.features = nn.Sequential(*self.layers)
        self.classifier = nn.Linear(num_hid, 10)

    def forward(self, x):
        features = self.features(x.view(-1, 3*32*32))
        output = self.classifier(features)
        return output

    def kaiming_init(self):
        for l in self.layers:
            name = type(l).__name__
            if name == 'ResLayerFC':
                l.kaiming_init()
            elif name == 'Linear':
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
            elif name == 'Linear':
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
            if name == 'ReLU' or name == 'Tanh':
                l.register_backward_hook(save_grad('first'))
            elif name == 'ResLayerFC':
                layer_name = 'reslayer_{}'.format(num_res_layer)
                l.register_hook(save_grad(layer_name))
                num_res_layer += 1
