import torch
from torch import nn
# from pytorch_ssim import SSIM

from kornia.losses.ssim import SSIM

# class MseSsimLoss(torch.nn.Module):
#     """
#     :deprecated nope, bug, don't use
#     """
#     def __init__(self):
#         super(MseSsimLoss, self).__init__()
#         self.mse = nn.MSELoss(reduction='sum')
#         self.ssim = SSIM(window_size=5)
#
#     def forward(self, input, target):
#         return self.mse(input, target) - self.ssim(input, target)


class MseSsimLoss(torch.nn.Module):
    """
    :deprecated nope, bug, don't use
    """
    def __init__(self, device):
        super(MseSsimLoss, self).__init__()
        self.device = device
        self.mse = nn.MSELoss(reduction='none').to(device)
        self.ssim = SSIM(window_size=11, reduction='none').to(device)

    def forward(self, input, target):
        # oneMat = torch.ones_like(input).to(self.device)
        # return (self.mse(input, target) + 0.01 * (oneMat - self.ssim(input, target))).sum()
        return (self.mse(input, target) + (self.ssim(input, target))).sum()
