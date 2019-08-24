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
    arg = parser.parse_args()
    return arg