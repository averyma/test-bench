import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
    
def load_dataset(_batch_size = 128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.ToTensor()])

    data_train = datasets.CIFAR10("./data", train=True, download = True, transform=transform_train)
    data_test = datasets.CIFAR10("./data", train=False, download = True, transform=transform_test)
    
    train_loader = DataLoader(data_train,
                            batch_size = _batch_size, 
                            shuffle = True, 
                            pin_memory = True,
                            num_workers = 1,
                            worker_init_fn = _init_fn)

    test_loader = DataLoader(data_test, 
                            batch_size = _batch_size,
                            shuffle = False,
                            pin_memory = True,
                            num_workers = 1,
                            worker_init_fn = _init_fn)

    return train_loader, test_loader

def _init_fn(manualSeed):
    np.random.seed(manualSeed)
    
def load_cifar10_rand_transform(attack_space, _batch_size = 128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.RandomAffine(degrees = attack_space[0], translate = [attack_space[1],attack_space[1]]),
        transforms.ToTensor()])

    data_train = datasets.CIFAR10("./data", train=True, download = True, transform=transform_train)
    data_test = datasets.CIFAR10("./data", train=False, download = True, transform=transform_test)
    
        
    train_loader = DataLoader(data_train,
                            batch_size = _batch_size, 
                            shuffle = True, 
                            pin_memory = True,
                            num_workers = 1,
                            worker_init_fn = _init_fn)

    test_loader = DataLoader(data_test, 
                            batch_size = _batch_size,
                            shuffle = False,
                            pin_memory = True,
                            num_workers = 1,
                            worker_init_fn = _init_fn)

    return train_loader, test_loader
