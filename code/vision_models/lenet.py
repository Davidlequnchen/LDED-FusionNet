'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
import numpy as np

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2) # 3 for RGB, change to 1 for grey-scale image and audio
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*6*6, 120) # input need to be determined
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 2) # 5 - output class

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, (2, 2))
        # out = out.view(out.size(0), -1)
        out = out.view(-1, self.num_flat_features(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)


def test():
    net = LeNet()
    print (net)
    # vision
    # x = torch.randn(1, 1, 28, 28)
    # y = net(x)
    # print(y.size())
    summary(net.cuda(), [(1, 32, 32)]) # image set

if __name__ == '__main__':
    test()