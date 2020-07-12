import torch
from torch.utils.data.dataloader import default_collate

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

def flatten_collate(batch):
    """

    :param batch:
    :return:
    """
    # remove null batch
    batch = list(filter(lambda x: x is not None, batch))

    # flattening array
    # in batch = [[s11, s12, s13], [s21,s22,s23], [s31,s32,s33]]
    # out batch = [s11, s12, s13, s21, s22, s23, s31, s32, s33]
    out_batch = [item for sample_list in batch for item in sample_list]
    return out_batch


