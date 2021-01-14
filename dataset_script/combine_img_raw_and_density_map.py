import numpy as np
from torchvision.io import write_jpeg, read_image
import torch
from torch import nn
import os

def overlay_img_with_density(img_path, density_map_path, output_path):
    """
    combine output density map with image to create the red heatmap overlay
    :param img_path:
    :param density_map_path: output .torch of density map
    :param output_path:
    :return:
    """
    img_tensor = read_image(img_path)
    density_map_tensor = torch.load(density_map_path)

    print(img_tensor.shape)
    print(density_map_tensor.shape)
    print(density_map_tensor.sum())
    density_map_tensor = torch.from_numpy(density_map_tensor).unsqueeze(dim=0).unsqueeze(dim=0)
    print("density_map_tensor.shape", density_map_tensor.shape)  # torch.Size([1, 1, 46, 82])
    upsampling_density_map_tensor = nn.functional.interpolate(density_map_tensor, scale_factor=8) / 64

    overlay_density_map = img_tensor.detach().clone()
    upsampling_density_map_tensor = (upsampling_density_map_tensor.squeeze(dim=0) / upsampling_density_map_tensor.max() * 255)
    overlay_density_map[0] = torch.clamp_max(img_tensor[0] + upsampling_density_map_tensor[0] * 2, max=255)

    write_jpeg(overlay_density_map.type(torch.uint8), output_path, quality=100)

def single_image_case():
    density = "/data/my_crowd_image/tmp/PRED_IMG_697.jpg.torch"
    img = "/data/my_crowd_image/video_bike_q100/IMG_697.jpg"
    output_path = "/data/my_crowd_image/tmp/PRED_OVERLAY_IMG_697.jpg"
    overlay_img_with_density(img, density, output_path)

def convert_folder():
    density_folder = "/data/my_crowd_image/bike_video_frame_404_dccnn_t4/"
    img_folder = "/data/my_crowd_image/video_bike_q100/"
    output_folder = "/data/my_crowd_image/overlay_bike_video_frame_404_dccnn_t4/"
    count = 0
    for img_name in os.listdir(img_folder):
        img_dir = os.path.join(img_folder, img_name)
        density_dir = os.path.join(density_folder, "PRED_" + img_name +".torch")
        output_dir = os.path.join(output_folder, "PRED_OVERLAY_" + img_name)
        overlay_img_with_density(img_dir, density_dir, output_dir)
        print("done " + img_name)
        count+=1
        print("count ", count)

if __name__ == "__main__":
    convert_folder()