import torch.nn as nn
import torch
from torchvision import models
import numpy as np

class PACNN(nn.Module):
    def __init__(self):
        super(PACNN, self).__init__()
        self.backbone =  models.vgg16(pretrained=True).features
        self.de1net = self.backbone[0:23]
        self.de1_11 = nn.Conv2d(512, 1, kernel_size=1)
        self.de2net = self.backbone[0:30]
        self.de2_11 = nn.Conv2d(512, 1, kernel_size=1)

        list_vgg16 = list(self.backbone)
        conv6_1_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        list_vgg16.append(conv6_1_1)
        self.de3net = nn.Sequential(*list_vgg16)
        self.de3_11 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        de1 = self.de1_11((self.de1net(x)))
        de2 = self.de2_11((self.de2net(x)))
        de3 = self.de3_11((self.de3net(x)))
        return de1, de2, de3

if __name__ == "__main__":
    # backbone = models.vgg19(pretrained=True)
    # print(backbone)
    # de1net = backbone[0:16]
    # de2net = backbone[0:23]
    # de3net = backbone[0:26]
    net = PACNN()
    print(net.de1net)
    img = torch.rand(1, 3, 320, 320)
    de1, de2, de3 = net(img)
    print(de1.size())
    print(de2.size())
    print(de3.size())
    # net = PACNN()
    # print(net)
