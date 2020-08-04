#  sha shanghaitech_keepfull is not convergent
from args_util import real_args_parse
from data_flow import get_train_val_list, get_dataloader, create_training_image_list
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, MeanAbsoluteError, MeanSquaredError
from crowd_counting_error_metrics import CrowdCountingMeanAbsoluteError, CrowdCountingMeanSquaredError
import torch
from torch import nn
import torch.nn.functional as F
from models import CSRNet,PACNN
import os
import cv2
from torchvision import datasets, transforms
from data_flow import ListDataset
import pytorch_ssim

from hard_code_variable import HardCodeVariable
from data_util import ShanghaiTechDataPath
from visualize_util import save_img, save_density_map


def visualize_shanghaitech_keepfull():
    HARD_CODE = HardCodeVariable()
    shanghaitech_data = ShanghaiTechDataPath(root=HARD_CODE.SHANGHAITECH_PATH)
    shanghaitech_data_part_a_train = shanghaitech_data.get_a().get_train().get()
    saved_folder = "visualize/debug_dataloader_shanghaitech"
    os.makedirs(saved_folder, exist_ok=True)
    train_list, val_list = get_train_val_list(shanghaitech_data_part_a_train, test_size=0.2)
    test_list = None
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="shanghaitech_keepfull", visualize_mode=True,
                                                           debug=True)

    # do with train loader
    train_loader_iter = iter(train_loader)
    for i in range(10):
        img, label, count = next(train_loader_iter)
        save_img(img, os.path.join(saved_folder, "train_img" + str(i) +".png"))
        save_path = os.path.join(saved_folder, "train_label" + str(i) +".png")
        save_density_map(label.numpy()[0][0], save_path)
        print("saved " + save_path)

if __name__ == "__main__":
    visualize_shanghaitech_keepfull()