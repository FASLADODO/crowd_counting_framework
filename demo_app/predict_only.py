import os
import torch
from data_flow import get_predict_dataloader
from models.dccnn import DCCNN
from visualize_util import save_density_map_normalize, save_density_map

if __name__ == "__main__":
    """
    predict all in folder 
    output into another folder 
    output density map and count in csv
    """
    NAME="bao2"
    # INPUT_FOLDER = "/data/ShanghaiTech/part_B/test_data/images/"
    INPUT_FOLDER = "/home/tt/Downloads/bao2"
    OUTPUT_FOLDER = "/data/apps/tmp"
    MODEL = "/home/tt/project/C-3-folder/trained_model/adamw1_bigtail13i_t1_shb_checkpoint_valid_mae=-7.574910521507263.pth"
    input_list = [os.path.join(INPUT_FOLDER, dir) for dir in os.listdir(INPUT_FOLDER)]
    loader = get_predict_dataloader(input_list)
    loaded_file = torch.load(MODEL)
    model = DCCNN()
    model.load_state_dict(loaded_file['model'])
    model.eval()
    os.mkdir(os.path.join(OUTPUT_FOLDER, NAME))
    log_file = open(os.path.join(OUTPUT_FOLDER, NAME, NAME +".log"), 'w')
    limit_count = 100
    count = 0
    for img, info in loader:
        if count > limit_count:
            break
        predict_name = "PRED_" + info["name"][0]

        predict_path = os.path.join(OUTPUT_FOLDER, NAME, predict_name)
        pred = model(img)
        pred = pred.detach().numpy()[0][0]
        pred_count = pred.sum()
        log_line = info["name"][0] + "," + str(pred_count.item()) +"\n"
        log_file.write(log_line)
        save_density_map(pred, predict_path)
        print("save to ", predict_path)
        count += 1
    log_file.close()
