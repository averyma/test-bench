import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class c1(nn.Module):
    '''c1 implemented in "Hessian-based analysis of Large batch trainig and robustness to adversaries"'''
    def __init__(self):
        super(c1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, 
                               out_channels = 64, 
                               kernel_size = 5, 
                               stride = 2,
                               padding = 18)

        self.conv2 = nn.Conv2d(in_channels = 64, 
                               out_channels = 64, 
                               kernel_size = 5, 
                               stride = 2,
                               padding = 7)

        self.mp = nn.MaxPool2d(3,3)
        self.bn = nn.BatchNorm2d(64)
        # self.bn = nn.BatchNorm2d(64, track_running_stats = False)

        self.fc1 = nn.Linear(64*3*3, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192,10)
        self.activation = nn.ReLU()

    def per_image_standardization(self, x):
        _dim = x.shape[1] * x.shape[2] * x.shape[3]
        mean = torch.mean(x, dim=(1,2,3), keepdim = True)
        stddev = torch.std(x, dim=(1,2,3), keepdim = True)
        adjusted_stddev = torch.max(stddev, (1./np.sqrt(_dim)) * torch.ones_like(stddev))
        return (x - mean) / adjusted_stddev

    def forward(self, x):
        x = self.per_image_standardization(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.mp(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.mp(x)
        x = self.bn(x)
        x = x.view(-1, 64*3*3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class c2(nn.Module):
    """mnist implemented in zico kolter's paper for wasserstein distance"""
    def __init__(self):
        super(c2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 16, 
                               kernel_size = 4, 
                               stride = 1,
                               padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 16, 
                               out_channels = 32, 
                               kernel_size = 4, 
                               stride = 1,
                               padding = 1)
        
        self.fc1 = nn.Linear(32*26*26, 100)
        self.fc2 = nn.Linear(100, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.view(-1, 32*26*26)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class c3(nn.Module):
    '''c3 implemented in "Hessian-based analysis of Large batch trainig and robustness to adversaries"'''
    def __init__(self, in_dim):
        super(c3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, 
                               out_channels = 64, 
                               kernel_size = 3, 
                               stride = 1,
                               padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 64, 
                               out_channels = 64, 
                               kernel_size = 3, 
                               stride = 1,
                               padding = 1)

        self.conv3 = nn.Conv2d(in_channels = 64, 
                               out_channels = 128, 
                               kernel_size = 3, 
                               stride = 1,
                               padding = 1)

        self.conv4 = nn.Conv2d(in_channels = 128, 
                               out_channels = 128, 
                               kernel_size = 3, 
                               stride = 1,
                               padding = 1)

        self.fc1 = nn.Linear(128*32*32, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = x.view(-1, 128*32*32)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class c4(nn.Module):
    """basically the same network design as c2, with the only difference being the CELU activation instead of RELU"""
    def __init__(self):
        super(c4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 16, 
                               kernel_size = 4, 
                               stride = 1,
                               padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 16, 
                               out_channels = 32, 
                               kernel_size = 4, 
                               stride = 1,
                               padding = 1)
        
        self.fc1 = nn.Linear(32*26*26, 100)
        self.fc2 = nn.Linear(100, 10)
        self.activation = nn.CELU()
        # self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.view(-1, 32*26*26)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class c5(nn.Module):
    """Implementation of FashionSimpleNet in this git repo, they claim to have 92% accuracy
	https://github.com/kefth/fashion-mnist/blob/master/model.py """
    def __init__(self):
        super(c5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, padding=1), # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return x

class c6(nn.Module):
    """basically the same network design as c2
       difference is that activation is ELU instead of relu
    """
    def __init__(self):
        super(c6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 16, 
                               kernel_size = 4, 
                               stride = 1,
                               padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 16, 
                               out_channels = 32, 
                               kernel_size = 4, 
                               stride = 1,
                               padding = 1)
        
        self.fc1 = nn.Linear(32*26*26, 100)
        self.fc2 = nn.Linear(100, 10)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.view(-1, 32*26*26)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class c7(nn.Module):
    """basically the same network design as c2
       difference is that activation is tanh instead of relu
    """
    def __init__(self):
        super(c7, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 16, 
                               kernel_size = 4, 
                               stride = 1,
                               padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 16, 
                               out_channels = 32, 
                               kernel_size = 4, 
                               stride = 1,
                               padding = 1)
        
        self.fc1 = nn.Linear(32*26*26, 100)
        self.fc2 = nn.Linear(100, 10)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.view(-1, 32*26*26)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class c8(nn.Module):
    """higher capacity model compare to c2, where the first out channel increased from 16->32, 
       and then the second layer, the out channel increased from 32->64
    """
    def __init__(self):
        super(c8, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 32, 
                               kernel_size = 3, 
                               stride = 1,
                               padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 32, 
                               out_channels = 64, 
                               kernel_size = 5, 
                               stride = 1,
                               padding = 1)
        
        self.fc1 = nn.Linear(64*26*26, 100)
        self.fc2 = nn.Linear(100, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.view(-1, 64*26*26)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class c9(nn.Module):
    def __init__(self):
        super(c9, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 8, 
                               kernel_size = 3, 
                               stride = 1,
                               padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 8, 
                               out_channels = 16, 
                               kernel_size = 3, 
                               stride = 1,
                               padding = 1)
        
        self.conv3 = nn.Conv2d(in_channels = 16, 
                               out_channels = 32, 
                               kernel_size = 5, 
                               stride = 1,
                               padding = 1)
        
        self.fc1 = nn.Linear(32*26*26, 100)
        self.fc2 = nn.Linear(100, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = x.view(-1, 32*26*26)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class c10(nn.Module):
    def __init__(self):
        super(c10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 32, 
                               kernel_size = 5, 
                               stride = 1,
                               padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 32, 
                               out_channels = 64, 
                               kernel_size = 5, 
                               stride = 1,
                               padding = 1)
        
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.activation = nn.ReLU()
        self.mp = nn.MaxPool2d(2,2, padding = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.mp(x)
        x = x.view(-1, 64*7*7)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
class c11(nn.Module):
    """
    https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
    This is a modified simple cnn arch taken from kera repo.
    Two major differences: 
    1. I removed the dropout layer: I tested both with/without dropout, and got higher
    clean accuracy without dropout
    2. I added the per_image_standardization.

    clean train: train 100 epochs with sgd, init lr of 0.01, decay at 50, 75, momentum of 0.9
    around 83-84% accuracy at epoch 50.
    """
    def __init__(self):
        super(c11, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def per_image_standardization(self, x):
        _dim = x.shape[1] * x.shape[2] * x.shape[3]
        mean = torch.mean(x, dim=(1,2,3), keepdim = True)
        stddev = torch.std(x, dim=(1,2,3), keepdim = True)
        adjusted_stddev = torch.max(stddev, (1./np.sqrt(_dim)) * torch.ones_like(stddev))
        return (x - mean) / adjusted_stddev

    def forward(self, x):
        x = self.per_image_standardization(x)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
#         x = F.dropout2d(x, training=self.training, p=0.25)
        
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
#         x = F.dropout2d(x, training=self.training, p=0.25)

        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training, p=0.5)
        x = self.fc2(x)
        return x
    
    def zero_activations(self, x, zero_act, non_zero_act, total_act, threshold):
        total_act += x.numel()
        zero_act += (x <= threshold).sum().item()
        non_zero_act += (x > threshold).sum().item()
        return zero_act, non_zero_act, total_act
    
    def return_non_zero_activations(self, x, threshold):
        zero_act = 0
        non_zero_act = 0
        total_act = 0
        
        x = self.per_image_standardization(x)
        x = F.relu(self.conv1(x))
        zero_act, non_zero_act, total_act = self.zero_activations(x, zero_act, non_zero_act, total_act, threshold)
        x = F.relu(self.conv2(x))
        zero_act, non_zero_act, total_act = self.zero_activations(x, zero_act, non_zero_act, total_act, threshold)
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        zero_act, non_zero_act, total_act = self.zero_activations(x, zero_act, non_zero_act, total_act, threshold)
        x = F.relu(self.conv4(x))
        zero_act, non_zero_act, total_act = self.zero_activations(x, zero_act, non_zero_act, total_act, threshold)
        x = self.pool(x)

        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        zero_act, non_zero_act, total_act = self.zero_activations(x, zero_act, non_zero_act, total_act, threshold)
        x = self.fc2(x)

        return zero_act, non_zero_act, total_act