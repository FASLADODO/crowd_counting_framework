from torchvision.io.video import read_video, write_video
from torchvision.io import write_png
import os


video_path = "/home/tt/Videos/VID_20201202_133703_090.mp4"
out_path = "../visualize/vid"


if __name__ == "__main__":
    v, a, info = read_video("/home/tt/Videos/VID_20201202_133703_090.mp4", pts_unit='sec')
    print("v type", type(v))
    print("a type", type(a))
    print(v.shape)  # torch.Size([467, 1080, 1920, 3])
    single_frame = v[100]
    print(single_frame.shape)  # torch.Size([1080, 1920, 3])

    batch_size_one = single_frame.unsqueeze(0)
    print(batch_size_one.shape)

    # single_frame = single_frame.permute(2, 0, 1)  # to CHW
    # print(single_frame.shape)
    # file_out = os.path.join(out_path, "single_frame.png")
    # write_png(single_frame, file_out)
    # print("done write to ", file_out)from torchvision.io import write_png