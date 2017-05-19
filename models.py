import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import re
import numpy as np

class FFUnit(nn.Module):
    def __init__(self, din, n_hidden_units, n_hidden_layers, nonlinearity=nn.Tanh, final_linear=False):
        super(FFUnit, self).__init__()
        self.layers = [nn.Linear(din, n_hidden_units), nonlinearity()]
        for _ in xrange(n_hidden_layers-1):
            self.layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            self.layers.append(nonlinearity())
        if final_linear:
            self.layers.append(nn.Linear(n_hidden_units, n_hidden_units))
        self.seq = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.seq(x)

    def randomize(self, bias_sigma, weight_sigma):
        for p in self.parameters():
            if p.dim() == 1: # bias
                torch.randn(p.size(), out=p.data).mul_(bias_sigma)
            if p.dim() == 2: # weight
                torch.randn(p.size(), out=p.data).mul_(
                    weight_sigma / np.sqrt(p.size()[1]))

class ResUnit(nn.Module):
    '''din should be equal to n_hidden_units'''
    def __init__(self, din, n_hidden_units, n_hidden_layers, nonlinearity=nn.Tanh, final_linear=False, res2out=False):
        super(ResUnit, self).__init__()
        self.res = FFUnit(din, n_hidden_units, n_hidden_layers, nonlinearity=nn.Tanh, final_linear=False)
        if res2out:
            self.res2out = nn.Linear(n_hidden_units, n_hidden_units)

    def forward(self, x):
        if hasattr(self, 'res2out'):
            return x + self.res2out(self.res(x))
        else:
            return x + self.res(x)

    def randomize(self, bias_sigma, weight_sigma, res_bias_sigma=None, res_weight_sigma=None, depth=None):
        '''res_bias_sigma = sigma_b
        res_weight_sigma = sigma_w
        bias_sigma = sigma_a
        weight_sigma = sigma_w'''
        if depth is not None:
            self.res.randomize(bias_sigma, 2**(-depth))
        else:
            self.res.randomize(bias_sigma, weight_sigma)
        if hasattr(self, 'res2out'):
            assert(res_bias_sigma is not None and res_weight_sigma is not None)
            for p in self.res2out.parameters():
                if p.dim() == 1: # bias
                    torch.randn(p.size(), out=p.data).mul_(res_bias_sigma)
                if p.dim() == 2: # weight
                    torch.randn(p.size(), out=p.data).mul_(
                        res_weight_sigma / np.sqrt(p.size()[1]))

class ResNet(nn.Module):
    def __init__(self, n_layers, resunit_args, resunit_kwargs={}):
        super(ResNet, self).__init__()
        n_hidden_units = resunit_args[1]

        self.layers = [nn.Linear(3*32*32, n_hidden_units)] # input
        for _ in xrange(n_layers):
            self.layers.append(ResUnit(*resunit_args, **resunit_kwargs))
        self.layers.append(nn.Linear(n_hidden_units, 10)) # output
        self.seq = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.seq(x.view(-1, 3*32*32))

    def randomize(self, bias_sigma, weight_sigma, kw, use_depth=False):
        randomize(self.layers[0], bias_sigma, weight_sigma)
        randomize(self.layers[-1], bias_sigma, weight_sigma)
        if use_depth:
            for j,l in enumerate(self.layers[1:-1]):
                kw['depth'] = j+1
                l.randomize(bias_sigma, weight_sigma, **kw)
        else:
            for l in self.layers[1:-1]:
                l.randomize(bias_sigma, weight_sigma, **kw)

    def collectgrads(self):
        grads = []
        for p in self.layers[0].parameters():
            g = np.zeros(list(p.data.size()) + [len(self.layers)])
            grads.append(g)
        for il, l in enumerate(self.layers):
            for i, p in enumerate(l.parameters()):
                grads[i][..., il] = p.grad.data.numpy()
        return grads

def randomize(layer, bias_sigma, weight_sigma):
    for p in layer.parameters():
        if p.dim() == 1: # bias
            torch.randn(p.size(), out=p.data).mul_(bias_sigma)
        if p.dim() == 2: # weight
            torch.randn(p.size(), out=p.data).mul_(
                weight_sigma / np.sqrt(p.size()[1]))

class FFNet(nn.Module):
    def __init__(self, n_layers, ffunit_args, ffunit_kwargs={}):
        super(FFNet, self).__init__()
        n_hidden_units = ffunit_args[1]

        self.layers = [nn.Linear(3*32*32, n_hidden_units)] # input
        for _ in xrange(n_layers):
            self.layers.append(FFUnit(*ffunit_args, **ffunit_kwargs))
        self.layers.append(nn.Linear(n_hidden_units, 10)) # output
        self.seq = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.seq(x.view(-1, 3*32*32))

    def randomize(self, bias_sigma, weight_sigma):
        randomize(self.layers[0], bias_sigma, weight_sigma)
        randomize(self.layers[-1], bias_sigma, weight_sigma)
        for l in self.layers[1:-1]:
            l.randomize(bias_sigma, weight_sigma)

class AlphaPlus(nn.Module):
    def __init__(self, alpha=2./3):
        super(AlphaPlus, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        v = 1 + F.relu(x)
        out = (v.log() * self.alpha).exp() - 1.
        return out

def torch_version(activ):
    name = re.findall(r'<ufunc \'(.*)\'>', repr(activ))
    if not name:
        raise Exception('idk')
    else:
        return getattr(nn, name[0].capitalize())
