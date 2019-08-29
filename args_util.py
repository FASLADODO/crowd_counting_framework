"""
contain dummy args with config
helpfull for copy paste Kaggle
"""
import argparse


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


def real_args_parse():
    """
    this is not dummy
    if you are going to make all-in-one notebook, ignore this
    :return:
    """
    parser = argparse.ArgumentParser(description='CrowdCounting')
    parser.add_argument("--task_id", action="store", default="dev")
    parser.add_argument('-a', action="store_true", default=False)

    parser.add_argument('--input', action="store",  type=str)
    parser.add_argument('--output', action="store", type=str)
    parser.add_argument('--model', action="store", default="csrnet")

    # args with default value
    parser.add_argument('--lr', action="store", default=1e-8, type=float)
    parser.add_argument('--momentum', action="store", default=0.95, type=float)
    parser.add_argument('--decay', action="store", default=5*1e-3, type=float)
    parser.add_argument('--epochs', action="store", default=1, type=int)

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