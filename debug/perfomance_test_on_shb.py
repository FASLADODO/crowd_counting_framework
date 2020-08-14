from comet_ml import Experiment

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
from model_util import get_lr, BestMetrics
from ignite.contrib.handlers import PiecewiseLinear
import time
import sys

if __name__ == "__main__":
    torch.set_num_threads(2)  # 4 thread

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = meow_parse()
    print(args)

    DATA_PATH = args.input
    TRAIN_PATH = os.path.join(DATA_PATH, "train_data_train_split")
    VAL_PATH = os.path.join(DATA_PATH, "train_data_validate_split")
    TEST_PATH = os.path.join(DATA_PATH, "test_data")
    dataset_name = args.datasetname
    if dataset_name=="shanghaitech":
        print("will use shanghaitech dataset with crop ")
    elif dataset_name == "shanghaitech_keepfull":
        print("will use shanghaitech_keepfull")
    else:
        print("cannot detect dataset_name")
        print("current dataset_name is ", dataset_name)

    # create list
    train_list = create_image_list(TRAIN_PATH)
    val_list = create_image_list(VAL_PATH)
    test_list = create_image_list(TEST_PATH)
    train_loader, train_loader_eval, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name=dataset_name, batch_size=args.batch_size,
                                                                              train_loader_for_eval_check=True,
                                                                              cache=args.cache,
                                                                              pin_memory=args.pin_memory,
                                                                              test_size=20)

    print("len train_loader ", len(train_loader))

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
        print("error: you didn't pick a model")
        exit(-1)
    model = model.to(device)
    checkpoint = torch.load(args.load_model)
    model.load_state_dict(checkpoint["model"])

    with torch.no_grad():
        model.eval()
        for test_time in range(10):
            print("test " + str(test_time))
            s1 = time.perf_counter()
            for img, label in test_loader:
                pred = model(img.cuda())
            print("done")
            s2 = time.perf_counter()
            time1 = s2 - s1
            print("time " + str(s2 - s1))
            sys.stdout.flush()
        print("done all")

