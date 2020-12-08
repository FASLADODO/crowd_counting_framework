import os
import torch
from data_flow import get_predict_dataloader
from models.dccnn import DCCNN
from models.compact_cnn import CompactCNNV7
from visualize_util import save_density_map_normalize, save_density_map

if __name__ == "__main__":
    """
    predict all in folder 
    output into another folder 
    output density map and count in csv
    """
    NAME="adamw1_ccnnv7_t4_bike_prediction"
    # INPUT_FOLDER = "/data/ShanghaiTech/part_B/test_data/images/"
    INPUT_FOLDER = "/data/my_crowd_image/dataset_batch1245/mybikedata/test_data/images/"
    OUTPUT_FOLDER = "/data/my_crowd_image/dataset_batch1245/mybikedata/test_data/predicts/"
    # MODEL = "/data/save_model/adamw1_bigtail13i_t1_bike/adamw1_bigtail13i_t1_bike_checkpoint_valid_mae=-2.8629838943481447.pth"
    MODEL = "/data/save_model/adamw1_ccnnv7_t4_bike/adamw1_ccnnv7_t4_bike_checkpoint_valid_mae=-3.143752908706665.pth"
    input_list = [os.path.join(INPUT_FOLDER, dir) for dir in os.listdir(INPUT_FOLDER)]
    loader = get_predict_dataloader(input_list)
    loaded_file = torch.load(MODEL)
    model = CompactCNNV7()
    model.load_state_dict(loaded_file['model'])
    model.eval()
    os.makedirs(os.path.join(OUTPUT_FOLDER, NAME), exist_ok=True)
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
        torch.save(pred, predict_path+".torch")
        print("save to ", predict_path)
        count += 1
    log_file.close()
