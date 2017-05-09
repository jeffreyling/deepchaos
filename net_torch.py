import torch.nn as nn
import torch
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

    def randomize(self, bias_sigma, weight_sigma, res_bias_sigma=None, res_weight_sigma=None):
        '''res_bias_sigma = sigma_b
        res_weight_sigma = sigma_w
        bias_sigma = sigma_a
        weight_sigma = sigma_w'''
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
        self.layers = []
        for _ in xrange(n_layers):
            self.layers.append(ResUnit(*resunit_args, **resunit_kwargs))
        self.seq = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.seq(x)

    def randomize(self, *args, **kw):
        for l in self.layers:
            l.randomize(*args, **kw)

    def collectgrads(self):
        grads = []
        for p in self.layers[0].parameters():
            g = np.zeros(list(p.data.size()) + [len(self.layers)])
            grads.append(g)
        for il, l in enumerate(self.layers):
            for i, p in enumerate(l.parameters()):
                grads[i][..., il] = p.grad.data.numpy()
        return grads

def torch_version(activ):
    name = re.findall(r'<ufunc \'(.*)\'>', repr(activ))
    if not name:
        raise Exception('idk')
    else:
        return getattr(nn, name[0].capitalize())
