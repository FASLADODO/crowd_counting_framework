import h5py
import torch
import shutil
import numpy as np
import os


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar'):
    if not os.path.exists("saved_model"):
        os.makedirs("saved_model")
    full_file_name = os.path.join("saved_model", task_id + filename)
    torch.save(state, full_file_name)
    if is_best:
        shutil.copyfile(task_id + filename, task_id + 'model_best.pth.tar')
    return full_file_name


def calculate_padding(kernel_size, dilation):
    """
    https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338

    o = output
    p = padding
    k = kernel_size
    s = stride
    d = dilation

    :return:
    """
    k = kernel_size
    d = dilation
    p = -1 + k + (k-1)*(d-1)
    p = p/2
    return p


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class BestMetrics:
    def __init__(self, best_metric="mae"):
        """
        :param best_metric: whether mae or mse will be use to determine best metric
        """
        self.best = 9999
        self.cur_mae = 9999
        self.cur_mse = 9999
        self.best_metric = "mae"

    def checkAndRecord(self, mae, mse):
        # check best
        flag_best = False
        if self.best_metric == "mae":
            if mae < self.best:
                flag_best = True
                self.best = mae
        elif self.best_metric == "mse":
            if mse < self.best:
                flag_best = True
                self.best = mse
        if flag_best:
            self.cur_mae = mae
            self.cur_mse = mse
        return flag_best


if __name__ == "__main__":
    print(calculate_padding(kernel_size=3, dilation=4))
    print(calculate_padding(kernel_size=5, dilation=1))
    print(calculate_padding(kernel_size=7, dilation=1))
    print(calculate_padding(kernel_size=9, dilation=1))
    print(calculate_padding(kernel_size=3, dilation=1))

    print("-----compact dilated cnn -----------------")
    print(calculate_padding(kernel_size=5, dilation=3))
    print(calculate_padding(kernel_size=5, dilation=2))
    print(calculate_padding(kernel_size=5, dilation=1))
    print("---dilated 3x3 with dilated 2 4 6")
    print(calculate_padding(kernel_size=3, dilation=2)) # 2
    print(calculate_padding(kernel_size=3, dilation=4)) # 4
    print(calculate_padding(kernel_size=3, dilation=6)) # 6