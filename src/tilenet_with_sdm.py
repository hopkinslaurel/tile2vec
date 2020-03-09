'''Modified ResNet-18 in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from species_labels import *

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TileNet(nn.Module):
    def __init__(self, num_blocks, in_channels=4, z_dim=512):
        super(TileNet, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.in_planes = 64

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(self.z_dim, num_blocks[4], stride=2)
        self.layer6 = nn.Sequential(nn.Linear(self.z_dim, 1), nn.Sigmoid())  # add a weight layer 

    def _make_layer(self, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.avg_pool2d(x, 4)
        z = x.view(x.size(0), -1)
        return z

    def encode_sdm(self, x):
        z = self.encode(x)  # torch.Size([n_batch, z_dim]), e.g. [48, 32] 
        y = self.layer6(z).flatten()  # returns [n_batch], e.g. [48]
        return (z, y)

    def forward(self, x):
        #return self.encode(x)
        return self.encode_sdm(x)
    
    def triplet_loss(self, z_p, z_n, z_d, y, p_sdm, csv_writer_indv, epoch, idx, margin=0.1, l2=0):
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        y = torch.Tensor(y).cuda()
        #print("y")
        #print(y.type)
        #print(y)
        #print("p_sdm")
        #print(p_sdm.type)
        #print(p_sdm)
        l_sdm = y*p_sdm + (1-y)*(1-p_sdm)
        #print("l_sdm")
        #print(l_sdm.type)
        #print(l_sdm)
        loss = F.relu(l_n + l_d + margin + l_sdm)
        # prep data to be written out
        if csv_writer_indv is not None:
            n = [x.item() for x in l_n]
            d = [x.item() for x in l_d]
            nd = [x.item() for x in l_nd]
            _p_sdm = [x.item() for x in p_sdm]
            _y = [x.item() for x in y]
            sdm = [x.item() for x in l_sdm]
            out = [epoch, idx]
            csv_writer_indv.writerow(out + ['l_n'] + n)
            csv_writer_indv.writerow(out + ['l_d'] + d)
            csv_writer_indv.writerow(out + ['l_nd'] + nd)
            csv_writer_indv.writerow(out + ['p_sdm'] + _p_sdm)
            csv_writer_indv.writerow(out + ['y'] + _y)
            csv_writer_indv.writerow(out + ['l_sdm'] + sdm)
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd

    def loss(self, patch, neighbor, distant, triplet_idx, species, csv_writer_indv, epoch, idx, margin=0.1, l2=0):
        """
        Computes loss for each batch.
        """
        z_p, p_sdm = self.encode_sdm(patch)  #z_p.shape: [48,32], p_sdm.shape: [48]
        #print(p_sdm)
        z_n, z_d = (self.encode(neighbor), self.encode(distant))
        y = get_records(triplet_idx, species)
        #print(y)
        loss, l_n, l_d, l_nd = self.triplet_loss(z_p, z_n, z_d, y, p_sdm, csv_writer_indv, epoch, idx, margin=margin, l2=l2)
        return loss, l_n, l_d, l_nd


def make_tilenet(in_channels=4, z_dim=512):
    """
    Returns a TileNet for unsupervised Tile2Vec with the specified number of
    input channels and feature dimension.
    """
    num_blocks = [2, 2, 2, 2, 2]
    return TileNet(num_blocks, in_channels=in_channels, z_dim=z_dim)

#a = make_tilenet(3,32)
#a.cuda()
#print(a)
#summary(a,(5,50,50))
