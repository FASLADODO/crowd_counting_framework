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
