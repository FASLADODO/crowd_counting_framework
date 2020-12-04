import cv2
import torch
from torch import nn
path = "/data/apps/tmp/for_question_forum"
img_path = "/home/tt/Downloads/bao2/download.jpeg"
density_map_path = "/data/apps/tmp/for_question_forum/PRED_download.jpeg.torch"

img_tensor = cv2.imread(img_path)
print(type(img_tensor))
print(img_tensor.shape)

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

pad_density_map_tensor = torch.zeros((1, 3, img_tensor.shape[0], img_tensor.shape[1]))
pad_density_map_tensor[:, 0,:upsampling_density_map_tensor.shape[2], :upsampling_density_map_tensor.shape[3]] = upsampling_density_map_tensor
print(pad_density_map_tensor.shape)
# pad_density_map_tensor = (pad_density_map_tensor.squeeze(dim=0)/pad_density_map_tensor.max()*255)
pad_density_map_tensor = pad_density_map_tensor.squeeze(dim=0)/pad_density_map_tensor.max()
print(pad_density_map_tensor.shape)
pad_density_map_tensor_match = pad_density_map_tensor.permute(1,2,0)
print(pad_density_map_tensor_match.shape)
pad_density_map_tensor_match_np = pad_density_map_tensor_match.numpy()
print(pad_density_map_tensor_match_np.shape)
print(pad_density_map_tensor_match_np.dtype)
pad_density_map_tensor_match_np = pad_density_map_tensor_match_np.astype("uint8")
print(img_tensor.dtype)
print(pad_density_map_tensor_match_np.dtype)
print(pad_density_map_tensor_match_np[:,:,0:1].shape)
overlay_color = cv2.applyColorMap(pad_density_map_tensor_match_np[:,:,0], colormap=cv2.COLORMAP_JET)

cv2.imwrite("../visualize/pic/cv2_overlay_color.png", overlay_color)
