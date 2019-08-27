import torch.nn as nn
import torch
from torchvision import models


class PACNN(nn.Module):
    def __init__(self):
        super(PACNN, self).__init__()
        self.backbone =  models.vgg16(pretrained=True).features
        self.de1net = self.backbone[0:16]
        self.de1_11 = nn.Conv2d(256, 1, kernel_size=1)
        self.de2net = self.backbone[0:23]
        self.de2_11 = nn.Conv2d(512, 1, kernel_size=1)
        self.de3net = self.backbone[0:26]
        self.de3_11 = nn.Conv2d(512, 1, kernel_size=1)


if __name__ == "__main__":
    backbone = models.vgg16(pretrained=True).features
    # print(backbone)
    de1net = backbone[0:16]
    de2net = backbone[0:23]
    de3net = backbone[0:26]
    print(de3net)
