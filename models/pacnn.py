import torch.nn as nn
import torch

from torchvision import models
import numpy as np
import copy

# ssim lost function


class PACNN(nn.Module):
    def __init__(self):
        super(PACNN, self).__init__()
        self.backbone = models.vgg16(pretrained=True).features
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
        return de1.squeeze(0), de2.squeeze(0), de3.squeeze(0)


class PACNNWithPerspectiveMap(nn.Module):
    def __init__(self, perspective_aware_mode=False):
        super(PACNNWithPerspectiveMap, self).__init__()
        self.backbone = models.vgg16(pretrained=True).features
        self.de1net = self.backbone[0:23]

        self.de2net = self.backbone[0:30]


        list_vgg16 = list(self.backbone)
        self.conv6_1_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        list_vgg16.append(self.conv6_1_1)
        self.de3net = nn.Sequential(*list_vgg16)


        self.conv5_2_3_stack = copy.deepcopy(self.backbone[23:30])
        self.perspective_net = nn.Sequential(self.backbone[0:23], self.conv5_2_3_stack)


        # 1 1 convolution
        self.de1_11 = nn.Conv2d(512, 1, kernel_size=1)
        self.de2_11 = nn.Conv2d(512, 1, kernel_size=1)
        self.de3_11 = nn.Conv2d(512, 1, kernel_size=1)
        self.perspective_11 = nn.Conv2d(512, 1, kernel_size=1)

        # deconvolution upsampling
        self.up12 = nn.ConvTranspose2d(1, 1, 2, 2)
        self.up23 = nn.ConvTranspose2d(1, 1, 2, 2)
        self.up_perspective = nn.ConvTranspose2d(1, 1, 2, 2)

        # if true, use perspective aware
        # if false, use average
        self.perspective_aware_mode = perspective_aware_mode

    def forward(self, x):
        de1 = self.de1_11((self.de1net(x)))
        de2 = self.de2_11((self.de2net(x)))
        de3 = self.de3_11((self.de3net(x)))
        if self.perspective_aware_mode:
            pespective_w_s = self.perspective_11(self.perspective_net(x))
            pespective_w = self.up_perspective(pespective_w_s)
            # TODO: code more here
            de23 = pespective_w_s * de2 + (1 - pespective_w_s)*(de2 + self.up23(de3))
            de = pespective_w * de1 + (1 - pespective_w)*(de1 + self.up12(de23))
        else:
            de23 = (de2 + self.up23(de3))/2
            de = (de1 + self.up12(de23))/2
        return de

def count_param(net):
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    return pytorch_total_params

def parameter_count_test():
    net = PACNN()
    total_real = count_param(net)
    print("total real ", total_real)
    backbone = count_param(net.backbone)
    conv611 = count_param(net.conv6_1_1)
    de1_11 = count_param(net.de1_11)
    de2_11 = count_param(net.de2_11)
    de3_11 = count_param(net.de3_11)
    sum_of_part = backbone + de1_11 + de2_11 + de3_11 + conv611
    print(sum_of_part)

if __name__ == "__main__":
    parameter_count_test()
    # net = PACNN()
    # print(net.de1net)
    # img = torch.rand(1, 3, 320, 320)
    # de1, de2, de3 = net(img)
    # print(de1.size())
    # print(de2.size())
    # print(de3.size())