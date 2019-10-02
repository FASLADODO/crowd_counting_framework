from comet_ml import Experiment
from args_util import real_args_parse
from data_flow import get_train_val_list, get_dataloader, create_training_image_list
from crowd_counting_error_metrics import CrowdCountingMeanAbsoluteError, CrowdCountingMeanSquaredError
import torch
from torch import nn
import torch.nn.functional as F
from models import CSRNet, PACNN, PACNNWithPerspectiveMap
import os
import cv2
from torchvision import datasets, transforms
from data_flow import ListDataset
import pytorch_ssim
from time import time
from evaluator import MAECalculator

from model_util import save_checkpoint

# import apex
# from apex import amp

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="S3mM1eMq6NumMxk2QJAXASkUM",
                            project_name="pacnn-dev2", workspace="ttpro1995")

    args = real_args_parse()
    print(device)
    print(args)


    MODEL_SAVE_NAME = args.task_id
    MODEL_SAVE_INTERVAL = 5
    DATA_PATH = args.input
    DATASET_NAME = "shanghaitech"
    TOTAL_EPOCH = args.epochs
    PACNN_PERSPECTIVE_AWARE_MODEL = args.PACNN_PERSPECTIVE_AWARE_MODEL

    experiment.set_name(args.task_id)
    experiment.log_parameter("DATA_PATH", DATA_PATH)
    experiment.log_parameter("PACNN_PERSPECTIVE_AWARE_MODEL", PACNN_PERSPECTIVE_AWARE_MODEL)
    experiment.log_parameter("train", "train without p")
    experiment.log_parameter("momentum", args.momentum)
    experiment.log_parameter("lr", args.lr)

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
    net = PACNNWithPerspectiveMap(perspective_aware_mode=PACNN_PERSPECTIVE_AWARE_MODEL).to(device)
    criterion_mse = nn.MSELoss(size_average=False).to(device)
    criterion_ssim = pytorch_ssim.SSIM(window_size=5).to(device)

    optimizer = torch.optim.SGD(net.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    # Allow Amp to perform casts as required by the opt_level
    # net, optimizer = amp.initialize(net, optimizer, opt_level="O1", enabled=False)

    current_save_model_name = ""
    current_epoch = 0

    # load model
    load_model = args.load_model
    if len(load_model) > 0:
        checkpoint = torch.load(load_model)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current_epoch = checkpoint['e']
        print("load ", load_model, "  epoch ", str(current_epoch))
    else:
        print("new model")

    while current_epoch < TOTAL_EPOCH:
        experiment.log_current_epoch(current_epoch)
        current_epoch += 1
        print("start epoch ", current_epoch)
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
            d1_label = d1_label.to(device).unsqueeze(0)
            d2_label = d2_label.to(device).unsqueeze(0)
            d3_label = d3_label.to(device).unsqueeze(0)

            # forward pass

            d1, d2, d3, p_s, p, d = net(train_img.to(device))
            loss_1 = criterion_mse(d1, d1_label) + criterion_ssim(d1, d1_label)
            loss_2 = criterion_mse(d2, d2_label) + criterion_ssim(d2, d2_label)
            loss_3 = criterion_mse(d3, d3_label) + criterion_ssim(d3, d3_label)
            loss_d = criterion_mse(d, d1_label) + criterion_ssim(d, d1_label)
            loss = loss_d + loss_1 + loss_2 + loss_3

            if PACNN_PERSPECTIVE_AWARE_MODEL:
                # TODO: loss for perspective map here
                pass
            loss_d = criterion_mse(d, d1_label) + criterion_ssim(d, d1_label)
            loss += loss_d
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
            sample += 1
            counting += 1

            if counting % 100 == 0:
                avg_loss_ministep = loss_sum/sample
                print("counting ", counting, " -- avg loss ", avg_loss_ministep)
                experiment.log_metric("avg_loss_ministep", avg_loss_ministep)
            # if counting == 100:
            #     break
            # end dataloader loop

        end_time = time()
        avg_loss = loss_sum/sample
        epoch_time = end_time - start_time
        print("==END epoch ", current_epoch, " =============================================")
        print(epoch_time, avg_loss, sample)
        experiment.log_metric("epoch_time", epoch_time)
        experiment.log_metric("avg_loss_epoch", avg_loss)
        print("=================================================================")

        if current_epoch % MODEL_SAVE_INTERVAL == 0:
            current_save_model_name = save_checkpoint({
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'e': current_epoch,
                    'PACNN_PERSPECTIVE_AWARE_MODEL': PACNN_PERSPECTIVE_AWARE_MODEL
                    # 'amp': amp.state_dict()
            }, False, MODEL_SAVE_NAME+"_"+str(current_epoch)+"_")
            experiment.log_asset(current_save_model_name)
            print("saved ", current_save_model_name)

        # end 1 epoch

        # after epoch evaluate
        mae_calculator_d1 = MAECalculator()
        mae_calculator_d2 = MAECalculator()
        mae_calculator_d3 = MAECalculator()
        mae_calculator_final = MAECalculator()
        with torch.no_grad():
            for val_img, label in val_loader_pacnn:
                net.eval()
                # load data
                d1_label, d2_label, d3_label = label

                # forward pass
                d1, d2, d3, p_s, p, d = net(val_img.to(device))

                d1_label = d1_label.to(device)
                d2_label = d2_label.to(device)
                d3_label = d3_label.to(device)

                # score
                mae_calculator_d1.eval(d1.cpu().detach().numpy(), d1_label.cpu().detach().numpy())
                mae_calculator_d2.eval(d2.cpu().detach().numpy(), d2_label.cpu().detach().numpy())
                mae_calculator_d3.eval(d3.cpu().detach().numpy(), d3_label.cpu().detach().numpy())
                mae_calculator_final.eval(d.cpu().detach().numpy(), d1_label.cpu().detach().numpy())
            print("count ", mae_calculator_d1.count)
            print("d1_val ", mae_calculator_d1.get_mae())
            print("d2_val ", mae_calculator_d2.get_mae())
            print("d3_val ", mae_calculator_d3.get_mae())
            print("dfinal_val ", mae_calculator_final.get_mae())
            experiment.log_metric("d1_val", mae_calculator_d1.get_mae())
            experiment.log_metric("d2_val", mae_calculator_d2.get_mae())
            experiment.log_metric("d3_val", mae_calculator_d3.get_mae())
            experiment.log_metric("dfinal_val", mae_calculator_final.get_mae())


    #############################################
    # done training evaluate
    net = PACNNWithPerspectiveMap(PACNN_PERSPECTIVE_AWARE_MODEL).to(device)
    print(net)

    best_checkpoint = torch.load(current_save_model_name)
    net.load_state_dict(best_checkpoint['model'])

    # device = "cpu"
    # TODO d1_val  155.97279205322266
    # d2_val  35.46327234903971
    # d3_val  23.07176342010498
    # why d2 and d3 mse too low
    mae_calculator_d1 = MAECalculator()
    mae_calculator_d2 = MAECalculator()
    mae_calculator_d3 = MAECalculator()
    mae_calculator_final = MAECalculator()
    with torch.no_grad():
        for val_img, label in val_loader_pacnn:
            net.eval()
            # load data
            d1_label, d2_label, d3_label = label

            # forward pass
            d1, d2, d3, p_s, p, d = net(val_img.to(device))

            d1_label = d1_label.to(device)
            d2_label = d2_label.to(device)
            d3_label = d3_label.to(device)

            # score
            mae_calculator_d1.eval(d1.cpu().detach().numpy(), d1_label.cpu().detach().numpy())
            mae_calculator_d2.eval(d2.cpu().detach().numpy(), d2_label.cpu().detach().numpy())
            mae_calculator_d3.eval(d3.cpu().detach().numpy(), d3_label.cpu().detach().numpy())
            mae_calculator_final.eval(d.cpu().detach().numpy(), d1_label.cpu().detach().numpy())
        print("count ", mae_calculator_d1.count)
        print("d1_val ", mae_calculator_d1.get_mae())
        print("d2_val ", mae_calculator_d2.get_mae())
        print("d3_val ", mae_calculator_d3.get_mae())
        print("dfinal_val ", mae_calculator_final.get_mae())
        experiment.log_metric("d1_val", mae_calculator_d1.get_mae())
        experiment.log_metric("d2_val", mae_calculator_d2.get_mae())
        experiment.log_metric("d3_val", mae_calculator_d3.get_mae())
        experiment.log_metric("dfinal_val", mae_calculator_final.get_mae())


