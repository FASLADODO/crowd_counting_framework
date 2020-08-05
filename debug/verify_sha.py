import torch
from models.meow_experiment.ccnn_tail import BigTail11i, BigTail10i, BigTail12i, BigTail13i, BigTail14i, BigTail15i
from hard_code_variable import HardCodeVariable
from data_util import ShanghaiTechDataPath
from visualize_util import save_img, save_density_map
import os
import numpy as np
from data_flow import get_train_val_list, get_dataloader, create_training_image_list
import cv2
import argparse


def _parse():
    parser = argparse.ArgumentParser(description='verify_sha')
    parser.add_argument('--input', action="store",  type=str, default=HardCodeVariable().SHANGHAITECH_PATH_PART_A)
    parser.add_argument('--output', action="store", type=str, default="visualize/verify_dataloader_shanghaitech")
    parser.add_argument('--meta_data', action="store", type=str, default="data_info.txt")
    parser.add_argument('--datasetname', action="store", default="shanghaitech_keepfull_r50")
    arg = parser.parse_args()
    return arg


def img_name_to_int(img_name):
    # IMG_174.jpg
    return int(img_name.replace("IMG_","").replace(".jpg",""))


def visualize_evaluation_shanghaitech_keepfull(path=None,
                                               dataset="shanghaitech_keepfull_r50",
                                               output="visualize/verify_dataloader_shanghaitech",
                                               meta_data="data_info.txt"):
    HARD_CODE = HardCodeVariable()
    if path==None:
        shanghaitech_data = ShanghaiTechDataPath(root= HARD_CODE.SHANGHAITECH_PATH)
        shanghaitech_data_part_a_train = shanghaitech_data.get_a().get_train().get()
        path = shanghaitech_data_part_a_train
    saved_folder = output
    os.makedirs(saved_folder, exist_ok=True)
    train_list, val_list = get_train_val_list(path, test_size=0.2)
    test_list = None
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name=dataset, visualize_mode=True,
                                                           debug=True)

    # do with train loader
    train_loader_iter = iter(train_loader)
    f = open(meta_data, "w")
    total = len(train_loader)
    for i in range(len(train_loader)):
        img, label, debug_data = next(train_loader_iter)
        p_count = debug_data["p_count"]
        name = debug_data["name"][0]
        item_number = img_name_to_int(name)
        density_map_count = label.sum()
        log_str = str(item_number) + " " + str(density_map_count.item()) + " " + str(p_count.item())
        print(log_str)
        f.write(log_str+"\n")
        save_img(img, os.path.join(saved_folder, "train_img_" + str(item_number) + ".png"))
        save_path = os.path.join(saved_folder, "train_label_"  + str(item_number) + ".png")
        save_density_map(label.numpy()[0][0], save_path)
        print(str(i) + "/" + str(total))
    f.close()


if __name__ == "__main__":
    args = _parse()
    visualize_evaluation_shanghaitech_keepfull(args.input, args.datasetname, args.output, args.meta_data)