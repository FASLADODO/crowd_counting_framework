from torchvision.io.video import read_video, write_video
import torch
from args_util import meow_parse
from data_flow import get_predict_video_dataloader
from models import create_model
import os
from visualize_util import save_density_map_normalize, save_density_map


video_path = "/data/mybikedata/VID_20201204_134210_960.mp4"
video_path = "/home/tt/Videos/VID_20201202_133703_090.mp4"
OUTPUT_FOLDER = "/data/my_crowd_image/dataset_batch1245/video/predicts/"
MODEL = "/data/save_model/adamw1_bigtail13i_t1_bike/adamw1_bigtail13i_t1_bike_checkpoint_valid_mae=-2.6874878883361815.pth"
out_path = "../visualize/vid"
NAME="VID_20201202_133703_090_again"
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
    loader = get_predict_video_dataloader(input_path)
    single_frame = None
    model = create_model("BigTail13i")
    loaded_file = torch.load(MODEL)
    model.load_state_dict(loaded_file['model'])
    model = model.to(device)
    model.eval()
    os.makedirs(os.path.join(OUTPUT_FOLDER, NAME), exist_ok=True)
    log_file = open(os.path.join(OUTPUT_FOLDER, NAME, NAME + ".log"), 'w')
    count = 0
    for frame, info in loader:
        # print("meow")
        frame = frame.to(device)
        pred = model(frame)
        index = info["index"][0].item()
        predict_name = "PRED_" + str(index)
        predict_path = os.path.join(OUTPUT_FOLDER, NAME, predict_name)
        pred = model(frame)
        pred = pred.detach().cpu().numpy()[0][0]
        pred_count = pred.sum()
        log_line = str(index) + "," + str(pred_count.item()) +"\n"
        log_file.write(log_line)
        save_density_map(pred, predict_path)
        torch.save(pred, predict_path+".torch")
        print("save to ", predict_path)
        count += 1
    log_file.close()

    print(single_frame)
