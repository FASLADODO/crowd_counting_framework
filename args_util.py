"""
contain dummy args with config
helpfull for copy paste Kaggle
"""
import argparse
from hard_code_variable import HardCodeVariable

def make_args(gpu="0", task="task_one_"):
    """
    these arg does not have any required commandline arg (all with default value)
    :param train_json:
    :param test_json:
    :param pre:
    :param gpu:
    :param task:
    :return:
    """
    parser = argparse.ArgumentParser(description='PyTorch CSRNet')

    args = parser.parse_args()
    args.gpu = gpu
    args.task = task
    args.pre = None
    return args

class Meow():
    def __init__(self):
        pass


def make_meow_args(gpu="0", task="task_one_"):
    args = Meow()
    args.gpu = gpu
    args.task = task
    args.pre = None
    return args


def like_real_args_parse(data_input):
    args = Meow()
    args.input = data_input
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 120
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 4
    args.print_freq = 30


def context_aware_network_args_parse():
    """
    this is not dummy
    if you are going to make all-in-one notebook, ignore this
    :return:
    """
    parser = argparse.ArgumentParser(description='CrowdCounting Context Aware Network')
    parser.add_argument("--task_id", action="store", default="dev")
    parser.add_argument('-a', action="store_true", default=False)

    parser.add_argument('--input', action="store",  type=str, default=HardCodeVariable().SHANGHAITECH_PATH_PART_A)
    parser.add_argument('--output', action="store", type=str, default="saved_model/context_aware_network")
    parser.add_argument('--datasetname', action="store", default="shanghaitech_keepfull")

    # args with default value
    parser.add_argument('--load_model', action="store", default="", type=str)
    parser.add_argument('--lr', action="store", default=1e-8, type=float)
    parser.add_argument('--momentum', action="store", default=0.9, type=float)
    parser.add_argument('--decay', action="store", default=5*1e-3, type=float)
    parser.add_argument('--epochs', action="store", default=1, type=int)
    parser.add_argument('--test', action="store_true", default=False)


    arg = parser.parse_args()
    return arg


def my_args_parse():
    parser = argparse.ArgumentParser(description='CrowdCounting Context Aware Network')
    parser.add_argument("--task_id", action="store", default="dev")
    parser.add_argument('--note', action="store", default="write anything")

    parser.add_argument('--input', action="store",  type=str, default=HardCodeVariable().SHANGHAITECH_PATH_PART_A)
    parser.add_argument('--datasetname', action="store", default="shanghaitech_keepfull")

    # args with default value
    parser.add_argument('--load_model', action="store", default="", type=str)
    parser.add_argument('--lr', action="store", default=1e-8, type=float)
    parser.add_argument('--momentum', action="store", default=0.9, type=float)
    parser.add_argument('--decay', action="store", default=5*1e-3, type=float)
    parser.add_argument('--epochs', action="store", default=1, type=int)
    parser.add_argument('--batch_size', action="store", default=1, type=int,
                        help="only set batch_size > 0 for dataset with image size equal")
    parser.add_argument('--test', action="store_true", default=False)
    arg = parser.parse_args()
    return arg


def meow_parse():
    parser = argparse.ArgumentParser(description='CrowdCounting Context Aware Network')
    parser.add_argument("--task_id", action="store", default="dev")
    parser.add_argument("--model", action="store", default="dev")
    parser.add_argument('--note', action="store", default="write anything")

    parser.add_argument('--input', action="store",  type=str, default=HardCodeVariable().SHANGHAITECH_PATH_PART_A)
    parser.add_argument('--datasetname', action="store", default="shanghaitech_keepfull")

    # args with default value
    parser.add_argument('--load_model', action="store", default="", type=str)
    parser.add_argument('--lr', action="store", default=1e-8, type=float)
    parser.add_argument('--momentum', action="store", default=0.9, type=float)
    parser.add_argument('--decay', action="store", default=5*1e-3, type=float)
    parser.add_argument('--epochs', action="store", default=1, type=int)
    parser.add_argument('--batch_size', action="store", default=1, type=int,
                        help="only set batch_size > 0 for dataset with image size equal")
    parser.add_argument('--test', action="store_true", default=False)
    arg = parser.parse_args()
    return arg

def sanity_check_dataloader_parse():
    parser = argparse.ArgumentParser(description='Dataloader')
    parser.add_argument('--input', action="store",  type=str, default=HardCodeVariable().SHANGHAITECH_PATH_PART_A)
    parser.add_argument('--datasetname', action="store", default="shanghaitech_keepfull")
    arg = parser.parse_args()
    return arg


def real_args_parse():
    """
    this is not dummy
    if you are going to make all-in-one notebook, ignore this
    :return:
    """
    parser = argparse.ArgumentParser(description='CrowdCounting')
    parser.add_argument("--task_id", action="store", default="dev")
    parser.add_argument('-a', action="store_true", default=False)

    parser.add_argument('--input', action="store",  type=str, default=HardCodeVariable().SHANGHAITECH_PATH_PART_A)
    parser.add_argument('--output', action="store", type=str, default="saved_model")
    parser.add_argument('--model', action="store", default="pacnn")

    # args with default value
    parser.add_argument('--load_model', action="store", default="", type=str)
    parser.add_argument('--lr', action="store", default=1e-8, type=float)
    parser.add_argument('--momentum', action="store", default=0.9, type=float)
    parser.add_argument('--decay', action="store", default=5*1e-3, type=float)
    parser.add_argument('--epochs', action="store", default=1, type=int)
    parser.add_argument('--test', action="store_true", default=False)

    # pacnn setting only
    parser.add_argument('--PACNN_PERSPECTIVE_AWARE_MODEL', action="store", default=0, type=int)
    parser.add_argument('--PACNN_MUTILPLE_SCALE_LOSS', action="store", default=1, type=int,
                        help="1: compare each of  density map/perspective map scale with gt for loss."
                             "0: only compare final density map and final density perspective map")

    # args.original_lr = 1e-7
    # args.lr = 1e-7
    # args.batch_size = 1
    # args.momentum = 0.95
    # args.decay = 5 * 1e-4
    # args.start_epoch = 0
    # args.epochs = 120
    # args.steps = [-1, 1, 100, 150]
    # args.scales = [1, 1, 1, 1]
    # args.workers = 4
    # args.seed = time.time()
    # args.print_freq = 30

    arg = parser.parse_args()
    return arg