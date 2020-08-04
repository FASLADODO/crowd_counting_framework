import torch
from models.meow_experiment.ccnn_tail import BigTail11i, BigTail10i, BigTail12i, BigTail13i, BigTail14i, BigTail15i
from hard_code_variable import HardCodeVariable
from data_util import ShanghaiTechDataPath
from visualize_util import save_img, save_density_map
import os
import numpy as np
from data_flow import get_train_val_list, get_dataloader, create_training_image_list
import cv2

def visualize_evaluation_shanghaitech_keepfull(path=None):
    HARD_CODE = HardCodeVariable()
    if path==None:
        shanghaitech_data = ShanghaiTechDataPath(root= HARD_CODE.SHANGHAITECH_PATH)
        shanghaitech_data_part_a_train = shanghaitech_data.get_a().get_train().get()
        path = shanghaitech_data_part_a_train
    saved_folder = "visualize/verify_dataloader_shanghaitech"
    os.makedirs(saved_folder, exist_ok=True)
    train_list, val_list = get_train_val_list(path, test_size=0.2)
    test_list = None
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="shanghaitech_keepfull", visualize_mode=True,
                                                           debug=True)

    # do with train loader
    train_loader_iter = iter(train_loader)
    for i in range(len(train_loader)):
        img, label, count = next(train_loader_iter)
        save_img(img, os.path.join(saved_folder, "train_img_" + str(i) +".png"))
        save_path = os.path.join(saved_folder, "train_label_"  + str(i) +".png")
        save_density_map(label.numpy()[0][0], save_path)


visualize_evaluation_shanghaitech_keepfull()

