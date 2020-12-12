from torchvision.io.video import read_video, write_video
import torch
from args_util import meow_parse
from data_flow import PredictVideoDataset
from models import create_model
import os


video_path = "/data/mybikedata/VID_20201204_134210_960.mp4"
out_path = "../visualize/vid"

if __name__ == "__main__":
    # n_thread = int(os.environ['OMP_NUM_THREADS'])
    # torch.set_num_threads(n_thread)  # 4 thread
    # print("n_thread ", n_thread)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # args = meow_parse()
    # print(args)
    # input_path = args.input
    input_path = video_path
    loader = PredictVideoDataset(input_path)
    single_frame = None
    for frame in loader:
        # print("meow")
        print(frame.shape)
        # single_frame = frame

    print(single_frame)
