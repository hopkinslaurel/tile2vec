# Edits to Neal Jean/Sherrie Wang code:
# Based on yunjey/pytorch-tutorial/ CNN and
# Modified ResNet-18 in PyTorch.
# Reference:
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#    Deep Residual Learning for Image Recognition. arXiv:1512.03385
# Also looked at Zagoruyko image patch sim work

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, in_channels=4, z_dim=512):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.in_planes = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.Linear(1024,z_dim)
        )
        
    def encode(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        return x

    def forward(self, x):
        return self.encode(x)

    def triplet_loss(self, z_p, z_n, z_d, margin=0.1, l2=0):
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + margin)
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd

    def loss(self, patch, neighbor, distant, margin=0.1, l2=0):
        """
        Computes loss for each batch.
        """
        z_p, z_n, z_d = (self.encode(patch), self.encode(neighbor),
            self.encode(distant))
        return self.triplet_loss(z_p, z_n, z_d, margin=margin, l2=l2)


def make_cnn(in_channels=5, z_dim=512):
    """
    Returns a TileNet for unsupervised Tile2Vec with the specified number of
    input channels and feature dimension.
    """
    return CNN(in_channels=in_channels, z_dim=z_dim)

# a = make_cnn(5, 512)
# a.cuda()
# summary(a,(5,50,50))

