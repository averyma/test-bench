import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.transforms.functional import InterpolationMode
import ipdb
from typing import Any, Callable, List, Optional, Union, Tuple
import os
from PIL import Image
import math

def load_dataset(dataset, batch_size=128, workers=4, distributed=False):
    
    # default augmentation
    if dataset.startswith('cifar') or dataset == 'svhn':
        if dataset == 'cifar10':
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
        elif dataset == 'cifar100':
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]
        elif dataset == 'svhn':
            mean = [0.4376821, 0.4437697, 0.47280442]
            std = [0.19803012, 0.20101562, 0.19703614]

        transform_train = transforms.Compose([
            # using 0.75 has a similar effect as pad 4 and randcrop
            # April 4 commented because it seems to cause NaN in training
            # transforms.RandomResizedCrop(32, scale=(0.75, 1.0), interpolation=Image.BICUBIC), 
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
    elif dataset == 'imagenet':
        # mean/std obtained from: https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
        # detail: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/7
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset == 'dummy':
        pass
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    # load dataset
    if dataset == 'cifar10':
        data_train = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
        data_test = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        data_train = datasets.CIFAR100("./data", train=True, download=True, transform=transform_train)
        data_test = datasets.CIFAR100("./data", train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        data_train = datasets.SVHN("./data/SVHN", split='train', download = True, transform=transform_train)
        data_test = datasets.SVHN("./data/SVHN", split='test', download = True, transform=transform_test)
    elif dataset == 'dummy':
        data_train = datasets.FakeData(5000, (3, 224, 224), 1000, transforms.ToTensor())
        data_test = datasets.FakeData(1000, (3, 224, 224), 1000, transforms.ToTensor())
    elif dataset == 'imagenet':
        dataroot = '/scratch/ssd002/datasets/imagenet'
        traindir = os.path.join(dataroot, 'train')
        valdir = os.path.join(dataroot, 'val')
        data_train = datasets.ImageFolder(traindir,transform_train)
        data_test = datasets.ImageFolder(valdir,transform_test)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_test, 
                                                                    shuffle=False,
                                                                    drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)

    return train_loader, test_loader, train_sampler, val_sampler
