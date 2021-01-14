from torchvision.io.video import read_video, write_video
from torchvision.io.image import write_jpeg
import torch
from args_util import meow_parse
from data_flow import get_predict_video_dataloader
from models import create_model
import os
from visualize_util import save_density_map_normalize, save_density_map

VIDEO_PATH = "/home/tt/Videos/VID_20201204_133931_404.mp4"
OUTPUT_PATH = "/data/my_crowd_image/video_bike_q100"
v, a, info = read_video(VIDEO_PATH, pts_unit='sec')
print(info)
print(v.shape)
length = v.shape[0]
print(length)

count = 0
for i in range(length):
    # if (i% 20 == 0):
    frame = v[i]
    frame = frame.permute(2, 0, 1)
    file_out_path = os.path.join(OUTPUT_PATH, "IMG_" + str(i) + ".jpg")
    write_jpeg(frame, file_out_path, quality=100)
    print(file_out_path)