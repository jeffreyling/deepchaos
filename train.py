from __future__ import print_function
import argparse
import sys
import os
import shutil
import time
import pdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import ResNet, AlphaPlus, FFNet
from ResModel import ResConvNet
from ResModelFC import ResNetFC
from util import *

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
                    # help='path to dataset')
# parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    # help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--reset-optimizer', dest='reset_optimizer', action='store_true',
                    help='if checkpointing, do not use loaded optimizer')
parser.add_argument('--savefile', required=True, help='savefile name')
parser.add_argument('--model', type=str, default='resnet', help='options: resnet, ff')

parser.add_argument('--weight_sigma', type=float, default=1.0)
parser.add_argument('--bias_sigma', type=float, default=0.05)
parser.add_argument('--res_weight_sigma', type=float, default=0.25)
parser.add_argument('--res_bias_sigma', type=float, default=0.05)
parser.add_argument('--nonlinearity', type=str, default='relu')
parser.add_argument('--n_hidden_units', type=int, default=500)
parser.add_argument('--n_layers', type=int, default=2**4)
parser.add_argument('--res2out', action='store_true', default=False)
parser.add_argument('--use_depth', action='store_true', default=False)

parser.add_argument('--check_grad', action='store_true', default=False)

best_prec1 = 0
nonlinearities = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'alpha': AlphaPlus}

def main():
    global args, best_prec1
    args = parser.parse_args()
    print('args:', args)

    # create model
    nonlinearity = nonlinearities[args.nonlinearity]
    print("=> creating model {}".format(args.model))
    if args.model == 'resnet':
        res_args = [args.n_hidden_units, args.n_hidden_units, 1]
        res_kwargs = {'nonlinearity': nonlinearity, 'res2out': args.res2out}
        model = ResNet(args.n_layers, res_args, res_kwargs)
    elif args.model == 'ff':
        ff_args = [args.n_hidden_units, args.n_hidden_units, 1]
        ff_kwargs = {'nonlinearity': nonlinearity}
        model = FFNet(args.n_layers, ff_args, ff_kwargs)
    elif args.model == 'resconvnet':
        model = ResConvNet(args.n_layers, nonlinearity=nonlinearity)
    elif args.model == 'resnetfc':
        model = ResNetFC(args.n_hidden_units, args.n_layers, nonlinearity=nonlinearity)

    model = model.cuda()

    print("=> initializing weights")
    if args.model == 'resnet':
        rand_kwargs = {'res_weight_sigma': args.res_weight_sigma, 'res_bias_sigma': args.res_bias_sigma}
        model.randomize(args.bias_sigma, args.weight_sigma, rand_kwargs, use_depth=args.use_depth)
    elif args.model == 'ff':
        model.randomize(args.bias_sigma, args.weight_sigma)
    elif args.model == 'resconvnet' or args.model == 'resnetfc':
        if args.use_depth:
            print("initializing with exponential depth...")
            model.greg_init()
        else:
            model.kaiming_init()

    print("number of model parameters: {}".format(get_num_parameters(model)))

    if args.check_grad:
        print("do greg's grad norm thing")
        grad_inputs = []
        grad_outputs = []
        def save_grad(name):
            def hook(layer, grad_input, grad_output):
                grad_inputs.append((name, grad_input[0].data.norm(2, 1).mean())) # dim 1
                grad_outputs.append((name, grad_output[0].data.norm(2, 1).mean())) # dim 1
            return hook
        model.register_hook(save_grad)

        rand_input = Variable(torch.rand(64, 3, 32, 32).cuda(), requires_grad=True)
        output = model(rand_input)
        # output.register_hook(print)
        rand_grad = torch.randn(64, 10).cuda()
        output.backward(rand_grad, retain_variables=True)

        d0 = np.log(grad_inputs[0][1])
        L = len(grad_inputs)
        ys = [np.log(g[1]) - d0 for g in grad_inputs[1:]]
        xs = [np.log(i) for i in range(L-1, 0, -1)]
        # xs = range(L-1, 0, -1)

        import pickle
        # pickle.dump(grad_inputs, open('batchnorm_relu.p', 'wb'))
        grad_inputs_relu = pickle.load(open('batchnorm_relu.p', 'rb'))
        d0_relu = np.log(grad_inputs_relu[0][1])
        ys_relu = [np.log(g[1]) - d0_relu for g in grad_inputs_relu[1:]]

        relu_plot = plt.scatter(xs, ys_relu, color='blue', label='batchnorm+relu')
        tanh_plot = plt.scatter(xs, ys, color='orange', label='tanh')
        plt.title('Scatter of grad norm vs. depth')
        plt.legend(handles=[relu_plot, tanh_plot])
        plt.xlabel('log(depth)')
        plt.ylabel('log(grad L2 norm)')
        # plt.savefig('grad_norms_no_log.png')
        plt.savefig('grad_norms.png')

        pdb.set_trace()
        sys.exit(0)
    # below is all for training

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

   # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # normalize = transforms.Lambda(per_image_whiten)
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.ToTensor(),
                        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)
        if args.model == 'resconvnet':
            epochToLearningRate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=args.savefile)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint'):
    name = '{}.pth.tar'.format(filename)
    torch.save(state, name)
    if is_best:
        best_name = '{}_best.pth.tar'.format(filename)
        shutil.copyfile(name, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.95 ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def epochToLearningRate(optimizer, epoch):
    # From https://github.com/bgshih/cifar.torch/blob/master/train.lua#L119-L128
    lr = 0.001
    if epoch == 0:
        lr = 0.01
    elif epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
