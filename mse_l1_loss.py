import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class MSEL1Loss(_Loss):

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSEL1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction) + F.l1_loss(input, target, reduction=self.reduction)


class MSE4L1Loss(_Loss):
    """
    weight is 1 MSE 4 L1
    """

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSE4L1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction) + 4*F.l1_loss(input, target, reduction=self.reduction)