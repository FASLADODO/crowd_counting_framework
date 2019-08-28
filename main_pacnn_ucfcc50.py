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

if __name__ == "__main__":
   #  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)
    args = real_args_parse()
    print(args)
    DATA_PATH = args.input


    # create list
    train_list, val_list = get_train_val_list(DATA_PATH, test_size=0.2)
    test_list = None

    # create data loader
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="ucf_cc_50")
    train_loader_pacnn = torch.utils.data.DataLoader(
        ListDataset(train_list,
                    shuffle=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]),
                    ]),
                    train=True,
                    batch_size=1,
                    num_workers=4, dataset_name="ucf_cc_50_pacnn"),
        batch_size=1, num_workers=4)

    # create model
    net = PACNN()

    for train_img, label in train_loader_pacnn:
        d1_label, d2_label, d3_label = label
        d1, d2, d3 = net(train_img)
        print(d1.size())
        print(d2.size())
        print(d3.size())
        print("====")
        print(d1_label.size())
        print(d2_label.size())
        print(d3_label.size())
        print("done===done=====")