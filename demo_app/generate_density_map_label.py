import os
import torch
from data_flow import get_predict_dataloader
from models.dccnn import DCCNN
from models.compact_cnn import CompactCNNV7
from visualize_util import save_density_map_normalize, save_density_map
import h5py
import numpy as np
import cv2

if __name__ == "__main__":
    """
    predict all in folder 
    output into another folder 
    output density map and count in csv
    """
    NAME="ground-truth-visualization"
    # INPUT_FOLDER = "/data/ShanghaiTech/part_B/test_data/images/"
    INPUT_FOLDER = "/data/my_crowd_image/dataset_batch1245/mybikedata/train_data/images/"
    GT_FOLDER = "/data/my_crowd_image/dataset_batch1245/mybikedata/train_data/ground-truth-h5/"
    OUTPUT_FOLDER = "/data/my_crowd_image/dataset_batch1245/mybikedata/train_data/predicts/"
    input_list = [os.path.join(INPUT_FOLDER, dir) for dir in os.listdir(INPUT_FOLDER)]
    loader = get_predict_dataloader(input_list)
    os.makedirs(os.path.join(OUTPUT_FOLDER, NAME), exist_ok=True)
    log_file = open(os.path.join(OUTPUT_FOLDER, NAME, NAME +".log"), 'w')
    limit_count = 100
    count = 0
    for img, info in loader:
        if count > limit_count:
            break
        predict_name = "PRED_" + info["name"][0]
        gt_name = info["name"][0].replace('.jpg', '.h5')
        predict_path = os.path.join(OUTPUT_FOLDER, NAME, predict_name)
        gt_path = os.path.join(GT_FOLDER, gt_name)
        gt_file = h5py.File(gt_path, 'r')
        target_origin = np.asarray(gt_file['density'])
        target = target_origin
        target_factor = 8
        target1 = cv2.resize(target,
                             (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                             interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
        img_path = info['img_path']
        pred = target1
        # pred = model(img)
        # pred = pred.detach().numpy()[0][0]
        pred_count = pred.sum()
        log_line = info["name"][0] + "," + str(pred_count.item()) +"\n"
        log_file.write(log_line)
        save_density_map(pred, predict_path)
        torch.save(pred, predict_path+".torch")
        print("save to ", predict_path)
        count += 1
    log_file.close()
