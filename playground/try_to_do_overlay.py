from torchvision.io import write_png, read_image
import torch
from torch import nn

path = "/data/apps/tmp/for_question_forum"
img_path = "/home/tt/Downloads/bao2/download.jpeg"
density_map_path = "/data/apps/tmp/for_question_forum/PRED_download.jpeg.torch"

img_tensor = read_image(img_path)
density_map_tensor = torch.load(density_map_path)

print(img_tensor.shape)
print(density_map_tensor.shape)
print(density_map_tensor.sum())
density_map_tensor = torch.from_numpy(density_map_tensor).unsqueeze(dim=0).unsqueeze(dim=0)
# module = nn.UpsamplingBilinear2d(scale_factor=8)
# upsampling_density_map_tensor = module(density_map_tensor)
upsampling_density_map_tensor = nn.functional.interpolate(density_map_tensor, scale_factor=8)/64
print(upsampling_density_map_tensor.sum())
print(upsampling_density_map_tensor.shape)

pad_density_map_tensor = torch.zeros((1, 3, img_tensor.shape[1], img_tensor.shape[2]))
pad_density_map_tensor[:, 0,:upsampling_density_map_tensor.shape[2], :upsampling_density_map_tensor.shape[3]] = upsampling_density_map_tensor
print(pad_density_map_tensor.shape)
pad_density_map_tensor = (pad_density_map_tensor.squeeze(dim=0)/pad_density_map_tensor.max()*255)
# pad_density_map_tensor = pad_density_map_tensor.squeeze(dim=0)

print(img_tensor.dtype)
print(pad_density_map_tensor.dtype)

overlay_density_map = img_tensor.detach().clone()
overlay_density_map[0] = torch.clamp_max(img_tensor[0] + pad_density_map_tensor[0] * 2, max=255)

write_png(overlay_density_map.type(torch.uint8), "../visualize/pic/overlay.png")
write_png(pad_density_map_tensor.type(torch.uint8), "../visualize/pic/pad.png")
