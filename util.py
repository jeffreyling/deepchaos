import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets

def get_num_parameters(model):
    N = sum([p.numel() for p in model.parameters()])
    return N

def per_image_whiten(image):
    mean = image.mean(); stddev = image.std()
    adjusted_stddev = max(stddev, 1.0/np.sqrt(image.numel()))
    return (image - mean) / adjusted_stddev

def random_labels(data_loader, num_labels=10):
    dataset = data_loader.dataset
    if dataset.train:
        labels = dataset.train_labels
    else:
        labels = dataset.test_labels

    for i in range(len(labels)):
        labels[i] = np.random.choice(num_labels)

def load_model(model_path, model):
    assert os.path.isfile(model_path), 'no file found at {}'.format(model_path)
    print("=> loading model '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, checkpoint['epoch']))

def load_datasets(batch_size=1, shuffle=False):
    # load data
    normalize = transforms.Lambda(per_image_whiten)
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.CenterCrop(28),
                            transforms.ToTensor(),
                            normalize,
                        ])),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.CenterCrop(28),
                            transforms.ToTensor(),
                            normalize,
                        ])),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True)

    return train_loader, val_loader
