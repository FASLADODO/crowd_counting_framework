
from args_util import meow_parse, lr_scheduler_milestone_builder
from data_flow import get_dataloader, create_image_list
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.handlers import Checkpoint, DiskSaver, Timer
from crowd_counting_error_metrics import CrowdCountingMeanAbsoluteError, CrowdCountingMeanSquaredError, CrowdCountingMeanAbsoluteErrorWithCount, CrowdCountingMeanSquaredErrorWithCount
from visualize_util import get_readable_time
from mse_l1_loss import MSEL1Loss, MSE4L1Loss
import torch
from torch import nn
from models.meow_experiment.kitten_meow_1 import M1, M2, M3, M4
from models.meow_experiment.ccnn_tail import BigTailM1, BigTailM2, BigTail3, BigTail4, BigTail5, BigTail6, BigTail7, BigTail8, BigTail6i, BigTail9i
from models.meow_experiment.ccnn_tail import BigTail11i, BigTail10i, BigTail12i, BigTail13i, BigTail14i, BigTail15i
from models.meow_experiment.ccnn_head import H1, H2, H3, H3i, H4i
from models.meow_experiment.kitten_meow_1 import H1_Bigtail3
from models import CustomCNNv2, CompactCNNV7
from models.compact_cnn import CompactCNNV8, CompactCNNV9, CompactCNNV7i
import os
from visualize_util import save_img, save_density_map
from model_util import get_lr, BestMetrics
from ignite.contrib.handlers import PiecewiseLinear
import time
import sys
from hard_code_variable import HardCodeVariable
from data_util import ShanghaiTechDataPath
import argparse
import cv2
import numpy as np
import math
from data_flow import get_train_val_list, get_dataloader, create_training_image_list
"""
This file evaluation on SHB and get information on evaluation process
"""

"/data/ShanghaiTech/part_A/test_data"

def _parse():
    parser = argparse.ArgumentParser(description='evaluatiuon SHB')
    parser.add_argument('--input', action="store",  type=str, default=HardCodeVariable().SHANGHAITECH_PATH_PART_A)
    parser.add_argument('--output', action="store", type=str, default="visualize/verify_dataloader_shanghaitech")
    parser.add_argument('--load_model', action="store", type=str, default=None)
    parser.add_argument('--model', action="store", type=str, default="visualize/verify_dataloader_shanghaitech")
    parser.add_argument('--meta_data', action="store", type=str, default="data_info.txt")
    parser.add_argument('--datasetname', action="store", default="shanghaitech_keepfull_r50")
    arg = parser.parse_args()
    return arg


def visualize_evaluation_shanghaitech_keepfull(model, args):
    """

    :param model: model with param, if not model then do not output pred
    :param args:
    :return:
    """
    if model is not None:
        model = model.cuda()
        model.eval()
    saved_folder = args.output
    os.makedirs(saved_folder, exist_ok=True)
    train_list, val_list = get_train_val_list(args.input, test_size=0.2)
    test_list = create_image_list(args.input)
    train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="shanghaitech_keepfull_r50", visualize_mode=False,
                                                           debug=True)

    log_f = open(args.meta_data, "w")
    mae_s = 0
    mse_s = 0
    n = 0
    train_loader_iter = iter(train_loader)
    _, gt_density,_ = next(train_loader_iter)
    with torch.no_grad():
        for item in test_loader:
            img, gt_density, debug_info = item
            gt_count = debug_info["p_count"]
            file_name = debug_info["name"]
            print(file_name[0].split(".")[0])
            file_name_only = file_name[0].split(".")[0]
            save_path = os.path.join(saved_folder, "label_" + file_name_only +".png")
            save_pred_path = os.path.join(saved_folder, "pred_" + file_name_only +".png")
            save_density_map(gt_density.numpy()[0], save_path)
            if model is not None:
                pred = model(img.cuda())
                predicted_density_map = pred.detach().cpu().clone().numpy()
                predicted_density_map_enlarge = cv2.resize(np.squeeze(predicted_density_map[0][0]), (int(predicted_density_map.shape[3] * 8), int(predicted_density_map.shape[2] * 8)), interpolation=cv2.INTER_CUBIC) / 64
                save_density_map(predicted_density_map_enlarge, save_pred_path)
                print("pred " + save_pred_path + " value " + str(predicted_density_map.sum()))

                print("cont compare " + str(predicted_density_map.sum()) + " " + str(predicted_density_map_enlarge.sum()))
                print("shape compare " + str(predicted_density_map.shape) + " " + str(predicted_density_map_enlarge.shape))
                density_map_count = gt_density.detach().sum()
                pred_count = pred.detach().cpu().sum()
                pred_count_num = pred_count.item()

                error = abs(pred_count_num-gt_count_num)
            else:
                error = 0
            mae_s += error
            mse_s += error*error

            density_map_count_num = density_map_count.item()
            gt_count_num = gt_count.item()
            log_str = str(file_name_only) + " " + str(density_map_count_num) + " " + str(gt_count.item()) + " " + str(pred_count.item())
            print(log_str)
            log_f.write(log_str+"\n")
    log_f.close()
    mae = mae_s / n
    mse = math.sqrt(mse_s / n)
    print("mae ", mae)
    print("mse", mse)


if __name__ == "__main__":
    if __name__ == "__main__":
        torch.set_num_threads(2)  # 4 thread

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        args = _parse()
        print(args)



        # # create list
        # train_list = create_image_list(TRAIN_PATH)
        # val_list = create_image_list(VAL_PATH)
        # test_list = create_image_list(TEST_PATH)
        # train_loader, train_loader_eval, val_loader, test_loader = get_dataloader(train_list, val_list, test_list,
        #                                                                           dataset_name=dataset_name,
        #                                                                           batch_size=args.batch_size,
        #                                                                           train_loader_for_eval_check=True,
        #                                                                           cache=args.cache,
        #                                                                           pin_memory=args.pin_memory,
        #                                                                           test_size=1)



        # model
        model_name = args.model

        if model_name == "M1":
            model = M1()
        elif model_name == "M2":
            model = M2()
        elif model_name == "M3":
            model = M3()
        elif model_name == "M4":
            model = M4()
        elif model_name == "CustomCNNv2":
            model = CustomCNNv2()
        elif model_name == "BigTailM1":
            model = BigTailM1()
        elif model_name == "BigTailM2":
            model = BigTailM2()
        elif model_name == "BigTail3":
            model = BigTail3()
        elif model_name == "BigTail4":
            model = BigTail4()
        elif model_name == "BigTail5":
            model = BigTail5()
        elif model_name == "BigTail6":
            model = BigTail6()
        elif model_name == "BigTail6i":
            model = BigTail6i()
        elif model_name == "BigTail9i":
            model = BigTail9i()
        elif model_name == "BigTail10i":
            model = BigTail10i()
        elif model_name == "BigTail11i":
            model = BigTail11i()
        elif model_name == "BigTail12i":
            model = BigTail12i()
        elif model_name == "BigTail13i":
            model = BigTail13i()
        elif model_name == "BigTail14i":
            model = BigTail14i()
        elif model_name == "BigTail15i":
            model = BigTail15i()
        elif model_name == "BigTail7":
            model = BigTail7()
        elif model_name == "BigTail8":
            model = BigTail8()
        elif model_name == "H1":
            model = H1()
        elif model_name == "H2":
            model = H2()
        elif model_name == "H3":
            model = H3()
        elif model_name == "H3i":
            model = H3i()
        elif model_name == "H4i":
            model = H4i()
        elif model_name == "H1_Bigtail3":
            model = H1_Bigtail3()
        elif model_name == "CompactCNNV7":
            model = CompactCNNV7()
        elif model_name == "CompactCNNV7i":
            model = CompactCNNV7i()
        elif model_name == "CompactCNNV8":
            model = CompactCNNV8()
        elif model_name == "CompactCNNV9":
            model = CompactCNNV9()
        else:
            print('no model ')
            model = None
        if args.load_model is not None:
            model = model.to(device)
            checkpoint = torch.load(args.load_model)
            model.load_state_dict(checkpoint["model"])
        model.eval()
        visualize_evaluation_shanghaitech_keepfull(model, args)



