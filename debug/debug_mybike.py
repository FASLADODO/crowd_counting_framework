from args_util import meow_parse
from data_flow import get_dataloader, create_image_list
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.handlers import Checkpoint, DiskSaver, Timer
from crowd_counting_error_metrics import CrowdCountingMeanAbsoluteError, CrowdCountingMeanSquaredError, CrowdCountingMeanAbsoluteErrorWithCount, CrowdCountingMeanSquaredErrorWithCount
from visualize_util import get_readable_time

import torch
from torch import nn
from models.meow_experiment.kitten_meow_1 import M1, M2, M3, M4
from models.meow_experiment.ccnn_tail import BigTailM1, BigTailM2, BigTail3, BigTail4
from models.meow_experiment.ccnn_head import H1, H2
from models.meow_experiment.kitten_meow_1 import H1_Bigtail3
from models import CustomCNNv2, CompactCNNV7
import os
from model_util import get_lr, BestMetrics
"""
shanghaitech_more_random
"""


if __name__ == "__main__":
    DATA_PATH = "/data/my_crowd_image/dataset_batch1245/mybikedata"
    TRAIN_PATH = os.path.join(DATA_PATH, "train_data")
    VAL_PATH = os.path.join(DATA_PATH, "train_data")
    TEST_PATH = os.path.join(DATA_PATH, "test_data")

    # create list
    train_list = create_image_list(TRAIN_PATH)
    val_list = create_image_list(VAL_PATH)
    test_list = create_image_list(TEST_PATH)

    train_loader, train_loader_eval, val_loader, test_loader = get_dataloader(train_list, val_list, test_list,
                                                                              dataset_name="my_bike_non_overlap"
                                                                              , batch_size=1,
                                                                              debug=True,
                                                                              train_loader_for_eval_check=True)
    print(len(train_loader))
    print(len(val_loader))

    for all in val_loader:
        print(all[0].shape)