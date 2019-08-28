
import torch
import torch.nn.functional as F

img = torch.rand(1,128,128)
img2 = F.interpolate(img, scale_factor=(1,0.5,0.5), mode="bicubic")