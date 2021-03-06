from args_util import real_args_parse
from data_flow import get_train_val_list, get_dataloader, create_training_image_list

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

def visualize_ucf_cc_50_pacnn():
    HARD_CODE = HardCodeVariable()
    saved_folder = "visualize/test_dataloader"
    os.makedirs(saved_folder, exist_ok=True)
    DATA_PATH = HARD_CODE.UCF_CC_50_PATH
    train_list, val_list = get_train_val_list(DATA_PATH, test_size=0.2)
    test_list = None

    # create data loader
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="ucf_cc_50")
    train_loader_pacnn = torch.utils.data.DataLoader(
        ListDataset(train_list,
                    shuffle=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ]),
                    train=True,
                    batch_size=1,
                    num_workers=4, dataset_name="shanghaitech_pacnn", debug=True),
        batch_size=1, num_workers=4)

    img, label = next(iter(train_loader_pacnn))

    print(img.shape)
    save_img(img, os.path.join(saved_folder, "pacnn_loader_img.png"))
    save_density_map(label[0].numpy()[0], os.path.join(saved_folder,"pacnn_loader_density1.png"))
    save_density_map(label[1].numpy()[0], os.path.join(saved_folder,"pacnn_loader_density2.png"))
    save_density_map(label[2].numpy()[0], os.path.join(saved_folder,"pacnn_loader_density3.png"))
    print("count1 ", label[0].numpy()[0].sum())
    print("count2 ", label[1].numpy()[0].sum())
    print("count3 ", label[2].numpy()[0].sum())


def visualize_shanghaitech_keepfull():
    HARD_CODE = HardCodeVariable()
    shanghaitech_data = ShanghaiTechDataPath(root=HARD_CODE.SHANGHAITECH_PATH)
    shanghaitech_data_part_a_train = shanghaitech_data.get_a().get_train().get()
    saved_folder = "visualize/test_dataloader_shanghaitech"
    os.makedirs(saved_folder, exist_ok=True)
    train_list, val_list = get_train_val_list(shanghaitech_data_part_a_train, test_size=0.2)
    test_list = None
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="shanghaitech_keepfull", visualize_mode=True)

    # do with train loader
    train_loader_iter = iter(train_loader)
    for i in range(10):
        img, label = next(train_loader_iter)
        save_img(img, os.path.join(saved_folder, "train_img" + str(i) +".png"))
        save_density_map(label.numpy()[0][0], os.path.join(saved_folder, "train_label" + str(i) +".png"))


def visualize_shanghaitech_pacnn_with_perspective():
    HARD_CODE = HardCodeVariable()
    shanghaitech_data = ShanghaiTechDataPath(root=HARD_CODE.SHANGHAITECH_PATH)
    shanghaitech_data_part_a_train = shanghaitech_data.get_a().get_train().get()
    saved_folder = "visualize/test_dataloader"
    os.makedirs(saved_folder, exist_ok=True)
    DATA_PATH = HARD_CODE.SHANGHAITECH_PATH_PART_A
    train_list, val_list = get_train_val_list(shanghaitech_data_part_a_train, test_size=0.2)
    test_list = None

    # create data loader
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="ucf_cc_50")
    train_loader_pacnn = torch.utils.data.DataLoader(
        ListDataset(train_list,
                    shuffle=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ]),
                    train=True,
                    batch_size=1,
                    num_workers=4, dataset_name="shanghaitech_pacnn_with_perspective", debug=True),
        batch_size=1, num_workers=4)

    img, label = next(iter(train_loader_pacnn))

    print(img.shape)
    save_img(img, os.path.join(saved_folder, "pacnn_loader_img.png"))
    save_density_map(label[0].numpy()[0], os.path.join(saved_folder,"pacnn_loader_with_p_density1.png"))
    save_density_map(label[1].numpy()[0], os.path.join(saved_folder,"pacnn_loader_with_p_density2.png"))
    save_density_map(label[2].numpy()[0], os.path.join(saved_folder,"pacnn_loader_with_p_density3.png"))
    save_density_map(label[3].numpy()[0], os.path.join(saved_folder, "pacnn_loader_p_s_4.png"))
    save_density_map(label[4].numpy()[0], os.path.join(saved_folder, "pacnn_loader_p_5.png"))
    print("count1 ", label[0].numpy()[0].sum())
    print("count2 ", label[1].numpy()[0].sum())
    print("count3 ", label[2].numpy()[0].sum())
    print("count4 ", label[3].numpy()[0].sum())
    print("count5 ", label[4].numpy()[0].sum())

    print("s1 ", label[0].shape)
    print("s2 ", label[1].shape)
    print("s3 ", label[2].shape)
    print("s4 ", label[3].shape)
    print("s5 ", label[4].shape)


def visualize_shanghaitech_nonoverlap_downsample():
    HARD_CODE = HardCodeVariable()
    shanghaitech_data = ShanghaiTechDataPath(root=HARD_CODE.SHANGHAITECH_PATH)
    shanghaitech_data_part_a_train = shanghaitech_data.get_a().get_train().get()
    saved_folder = "visualize/test_dataloader"
    os.makedirs(saved_folder, exist_ok=True)
    DATA_PATH = HARD_CODE.SHANGHAITECH_PATH_PART_B
    train_list, val_list = get_train_val_list(shanghaitech_data_part_a_train, test_size=0.2)
    test_list = None

    # create data loader
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="shanghaitech_non_overlap_downsample",
                                                           visualize_mode=True)


    img, label = next(iter(train_loader))

    print(img.shape)
    save_img(img, os.path.join(saved_folder, "overlap_downsample_loader_1.png"))
    save_density_map(label[0].numpy()[0], os.path.join(saved_folder,"overlap_downsample_loader_with_p_density1.png"))

    print("count1 ", label.numpy()[0].sum())
    print("count2 ", label.numpy()[0].sum())
    print("count3 ", label.numpy()[0].sum())

    print("s1 ", label.shape)




if __name__ == "__main__":
    # visualize_shanghaitech_pacnn_with_perspective()
    visualize_shanghaitech_nonoverlap_downsample()