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
from time import time
from evaluator import MAECalculator

from model_util import save_checkpoint

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device = "cpu"
    print(device)
    args = real_args_parse()
    print(args)
    DATA_PATH = args.input
    DATASET_NAME = "shanghaitech"

    # create list
    if DATASET_NAME is "shanghaitech":
        TRAIN_PATH = os.path.join(DATA_PATH, "train_data")
        TEST_PATH = os.path.join(DATA_PATH, "test_data")
        train_list, val_list = get_train_val_list(TRAIN_PATH)
        test_list = create_training_image_list(TEST_PATH)
    elif DATASET_NAME is "ucf_cc_50":
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
                    num_workers=4, dataset_name="shanghaitech_pacnn"),
        batch_size=1, num_workers=4)

    val_loader_pacnn = torch.utils.data.DataLoader(
        ListDataset(val_list,
                    shuffle=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]),
                    ]),
                    train=False,
                    batch_size=1,
                    num_workers=4, dataset_name="shanghaitech_pacnn"),
        batch_size=1, num_workers=4)

    # create model
    net = PACNN().to(device)
    criterion_mse = nn.MSELoss(size_average=False).to(device)
    criterion_ssim = pytorch_ssim.SSIM(window_size=11).to(device)

    optimizer = torch.optim.SGD(net.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    for e in range(1):
        print("start epoch ", e)
        loss_sum = 0
        sample = 0
        start_time = time()
        counting = 0
        for train_img, label in train_loader_pacnn:
            net.train()
            # zero the parameter gradients
            optimizer.zero_grad()

            # load data
            d1_label, d2_label, d3_label = label
            d1_label = d1_label.to(device)
            d2_label = d2_label.to(device)
            d3_label = d3_label.to(device)

            # forward pass

            d1, d2, d3 = net(train_img.to(device))
            loss_1 = criterion_mse(d1, d1_label) + criterion_ssim(d1.unsqueeze(0), d1_label.unsqueeze(0))
            loss_2 = criterion_mse(d2, d2_label) + criterion_ssim(d2.unsqueeze(0), d2_label.unsqueeze(0))
            loss_3 = criterion_mse(d3, d3_label) + criterion_ssim(d3.unsqueeze(0), d3_label.unsqueeze(0))

            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            sample += 1
            optimizer.zero_grad()
            counting += 1
            if counting%10 ==0:
                print("counting ", counting, " -- avg loss", loss_sum/sample)
            # if counting == 100:
            #     break

        end_time = time()
        avg_loss = loss_sum/sample
        epoch_time = end_time - start_time
        print(epoch_time, avg_loss, sample)


        save_checkpoint({
            'state_dict': net.state_dict(),
        }, False, "test2")



    # evaluate

    best_checkpoint = torch.load("test2checkpoint.pth.tar")
    net = PACNN().to(device)
    print(net)
    net.load_state_dict(best_checkpoint['state_dict'])

    # device = "cpu"
    mae_calculator_d1 = MAECalculator()
    mae_calculator_d2 = MAECalculator()
    mae_calculator_d3 = MAECalculator()
    with torch.no_grad():
        for val_img, label in val_loader_pacnn:
            net.eval()
            # load data
            d1_label, d2_label, d3_label = label

            # forward pass
            d1, d2, d3 = net(val_img.to(device))

            d1_label = d1_label.to(device)
            d2_label = d2_label.to(device)
            d3_label = d3_label.to(device)

            # score
            mae_calculator_d1.eval(d1.cpu().detach().numpy(), d1_label.cpu().detach().numpy())
            mae_calculator_d2.eval(d2.cpu().detach().numpy(), d2_label.cpu().detach().numpy())
            mae_calculator_d3.eval(d3.cpu().detach().numpy(), d3_label.cpu().detach().numpy())
        print("count ", mae_calculator_d1.count)
        print("d1_val ", mae_calculator_d1.get_mae())
        print("d2_val ", mae_calculator_d2.get_mae())
        print("d3_val ", mae_calculator_d3.get_mae())


