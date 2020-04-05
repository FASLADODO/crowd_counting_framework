import torch.nn as nn
import torch
import collections
import torch.nn.functional as F

"""
let have 6 tail + 1 output layer

we always keep head of 
A REAL-TIME DEEP NETWORK FOR CROWD COUNTING
https://arxiv.org/pdf/2002.06515.pdf

"""


class BigTailM1(nn.Module):
    """
    """
    def __init__(self, load_weights=False):
        super(BigTailM1, self).__init__()
        self.model_note = "small big tail 512"
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c0 = nn.Conv2d(40, 512, 3, padding=2, dilation=2)
        self.c1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.c2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(512, 256, 3, padding=2, dilation=2)
        self.c4 = nn.Conv2d(256, 128, 3, padding=2, dilation=2)
        self.c5 = nn.Conv2d(128, 64, 3, padding=2, dilation=2)
        self.output = nn.Conv2d(64, 1, 1)

    def forward(self,x):
        x_red = self.max_pooling(F.relu(self.red_cnn(x), inplace=True))
        x_green = self.max_pooling(F.relu(self.green_cnn(x), inplace=True))
        x_blue = self.max_pooling(F.relu(self.blue_cnn(x), inplace=True))

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c0(x), inplace=True)

        x = F.relu(self.c1(x), inplace=True)

        x = F.relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c4(x), inplace=True)

        x = F.relu(self.c5(x), inplace=True)

        x = self.output(x)
        return x


class BigTailM2(nn.Module):
    """

    """
    def __init__(self, load_weights=False):
        super(BigTailM2, self).__init__()
        self.model_note = "small taill 100"
        self.red_cnn = nn.Conv2d(3, 10, 9, padding=4)
        self.green_cnn = nn.Conv2d(3, 14, 7, padding=3)
        self.blue_cnn = nn.Conv2d(3, 16, 5, padding=2)

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c0 = nn.Conv2d(40, 100, 3, padding=2, dilation=2)
        self.c1 = nn.Conv2d(100, 100, 3, padding=2, dilation=2)
        self.c2 = nn.Conv2d(100, 100, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(100, 50, 3, padding=2, dilation=2)
        self.c4 = nn.Conv2d(50, 25, 3, padding=2, dilation=2)
        self.c5 = nn.Conv2d(25, 10, 3, padding=2, dilation=2)
        self.output = nn.Conv2d(10, 1, 1)

    def forward(self,x):
        x_red = self.max_pooling(F.relu(self.red_cnn(x), inplace=True))
        x_green = self.max_pooling(F.relu(self.green_cnn(x), inplace=True))
        x_blue = self.max_pooling(F.relu(self.blue_cnn(x), inplace=True))

        x = torch.cat((x_red, x_green, x_blue), 1)
        x = F.relu(self.c0(x), inplace=True)

        x = F.relu(self.c1(x), inplace=True)

        x = F.relu(self.c2(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c3(x), inplace=True)
        x = self.max_pooling(x)

        x = F.relu(self.c4(x), inplace=True)

        x = F.relu(self.c5(x), inplace=True)

        x = self.output(x)
        return x


