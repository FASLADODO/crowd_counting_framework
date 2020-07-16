import torch
from torch.utils.data.dataloader import default_collate
import random

def my_collate(batch):  # batch size 4 [{tensor image, tensor label},{},{},{}] could return something like G = [None, {},{},{}]
    """
    collate that ignore None
    However, if all sample is None, we have problem, so, set batch size bigger
    https://stackoverflow.com/questions/57815001/pytorch-collate-fn-reject-sample-and-yield-another
    :param batch: list
    :return: list
    """
    batch = list(filter (lambda x:x is not None, batch)) # this gets rid of nones in batch. For example above it would result to G = [{},{},{}]
    # I want len(G) = 4
    # so how to sample another dataset entry?
    return torch.utils.data.dataloader.default_collate(batch)

def flatten_collate_broken(batch):
    """

    :param batch: tuple of (data, label)
    :return:
    """
    # remove null batch
    batch = list(filter(lambda x: x is not None, batch))

    # flattening array
    # in batch = [[s11, s12, s13], [s21,s22,s23], [s31,s32,s33]]
    # out batch = [s11, s12, s13, s21, s22, s23, s31, s32, s33]
    out_batch = [item for sample_list in batch for item in sample_list]
    return out_batch


def _flatten_collate(batch):
    """

    :param batch: tuple of (data, label) with type(data) == list, type(label) == list
    :return: flatten data, label
    """
    # remove null batch
    batch = list(filter(lambda x: x is not None, batch))

    # flattening array

    # more clarify version
    # out_batch = []
    # for data_pair in batch:
    #     for img, label in zip(*data_pair):
    #         out_batch.append((img, label))

    # python List Comprehensions
    out_batch = list([(img, label) for data_pair in batch for img, label in zip(*data_pair)])

    # shuffle data in batch
    # explain: dataset shuffle only shuffle index
    # each index (sample) generate multiple image which is not shuffle
    # so we have to shuffle them all
    random.shuffle(out_batch)

    return out_batch


def flatten_collate(batch):
    """

    :param batch: tuple of (data, label) with type(data) == list, type(label) == list
    :return: flatten data, label
    """
    # remove null batch
    batch1 = _flatten_collate(batch)
    out_batch = torch.utils.data.dataloader.default_collate(batch1)
    return out_batch