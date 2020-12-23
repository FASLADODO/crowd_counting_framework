from torchvision.io import write_jpeg, read_image
import torch
import cv2
import os
from tqdm import tqdm
import glob

def explore():
    img = read_image("/data/my_crowd_image/video_bike_q100/IMG_697.jpg")
    img2 = read_image("/data/my_crowd_image/video_bike_q100/IMG_697.jpg")
    print(img.shape)

    c = torch.cat([img.unsqueeze(0), img2.unsqueeze(0)])
    print(c.shape)

def create_video():

    # TODO
    img_prefix = '/data/my_crowd_image/overlay_bike_video_frame_404_dccnn_t4/PRED_OVERLAY_IMG_'
    video_name = '/data/my_crowd_image/overlay_bike_video_frame_404_dccnn_t4_video_29_99512746811621fps.avi'  # save as .avi
    # is changeable but maintain same h&w over all  frames
    width = 1920
    height = 1080
    # this fourcc best compatible for avi
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter(video_name, fourcc, 29.99512746811621, (width, height))

    for i in tqdm(range(2013)):
        img_dir = img_prefix + str(i) + ".jpg"
        x = cv2.imread(img_dir)
        video.write(x)

    # cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    create_video()