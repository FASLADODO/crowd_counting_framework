from __future__ import division

import torch
import torch.nn.functional as F
import math
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric

class CrowdCountingMeanAbsoluteError(Metric):
    """
    Calculates the mean absolute error.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_of_absolute_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred = output[0]
        y = output[1]
        pred_count = torch.sum(y_pred)
        true_count = torch.sum(y)
        absolute_errors = torch.abs(pred_count - true_count)
        self._sum_of_absolute_errors += torch.sum(absolute_errors).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanAbsoluteError must have at least one example before it can be computed.')
        return self._sum_of_absolute_errors / self._num_examples


class CrowdCountingMeanSquaredError(Metric):
    """
    Calculates the mean squared error.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_of_squared_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred = output[0]
        y = output[1]
        pred_count = torch.sum(y_pred)
        true_count = torch.sum(y)
        squared_errors = torch.pow(pred_count - true_count, 2)
        self._sum_of_squared_errors += torch.sum(squared_errors).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanSquaredError must have at least one example before it can be computed.')
        return math.sqrt(self._sum_of_squared_errors / self._num_examples)

###########################################


class CrowdCountingMeanAbsoluteErrorWithCount(Metric):
    """
    Calculates the mean absolute error.
    Compare directly with original count

    - `update` must receive output of the form `(y_pred, y, count)`.
    """
    def reset(self):
        self._sum_of_absolute_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred = output[0]
        true_count = output[1]
        pred_count = torch.sum(y_pred)
        # true_count = torch.sum(y)
        absolute_errors = torch.abs(pred_count - true_count)
        self._sum_of_absolute_errors += torch.sum(absolute_errors).item()
        self._num_examples += y_pred.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanAbsoluteError must have at least one example before it can be computed.')
        return self._sum_of_absolute_errors / self._num_examples


class CrowdCountingMeanSquaredErrorWithCount(Metric):
    """
    Calculates the mean squared error.
    Compare directly with original count

    - `update` must receive output of the form `(y_pred, y, count)`.
    """
    def reset(self):
        self._sum_of_squared_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred = output[0]
        true_count = output[1]
        pred_count = torch.sum(y_pred)
        # true_count = torch.sum(y)
        squared_errors = torch.pow(pred_count - true_count, 2)
        self._sum_of_squared_errors += torch.sum(squared_errors).item()
        self._num_examples += y_pred.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanSquaredError must have at least one example before it can be computed.')
        return math.sqrt(self._sum_of_squared_errors / self._num_examples)

####################
import piq

class CrowdCountingMeanSSIMabs(Metric):
    """
    Calculates ssim
    require package https://github.com/photosynthesis-team/piq
    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred = output[0]
        y = output[1]
        # y_pred = torch.clamp_min(y_pred, min=0.0)
        # y = torch.clamp_min(y, min=0.0)
        y = torch.abs(y)
        y_pred = torch.abs(y_pred)
        # print("CrowdCountingMeanSSIMabs ")
        # print("y_pred", y_pred.shape)
        # print("y", y.shape)

        y_pred = F.interpolate(y_pred, scale_factor=8)/64
        pad_density_map_tensor = torch.zeros((1, 1, y.shape[2], y.shape[3])).cuda()
        pad_density_map_tensor[:, 0, :y_pred.shape[2],:y_pred.shape[3]] = y_pred
        y_pred = pad_density_map_tensor

        # rig_y = torch.sum(y)
        # rig_y_pred = torch.sum(y_pred)
        # y_max = torch.max(y)
        # y_pred_max = torch.max(y_pred)
        # max_value = y_max
        ssim_metric = piq.ssim(y, y_pred, reduction="sum", data_range=max_value.item())
        # ssim_metric = torch.abs(rig_y - rig_y_pred)



        self._sum += ssim_metric.item()
        # we multiply because ssim calculate mean of each image in batch
        # we multiply so we will divide correctly

        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CrowdCountingMeanSSIM must have at least one example before it can be computed.')
        return self._sum / self._num_examples


class CrowdCountingMeanPSNRabs(Metric):
    """
    Calculates ssim
    require package https://github.com/photosynthesis-team/piq
    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred = output[0]
        # y_pred = torch.clamp_min(y_pred, min=0.0)
        y = output[1]
        # y = torch.clamp_min(y, min=0.0)
        # y = torch.abs(y)
        # y_pred = torch.abs(y_pred)
        # print("CrowdCountingMeanPSNRabs ")
        # print("y_pred", y_pred.shape)
        # print("y", y.shape)

        y_pred = F.interpolate(y_pred, scale_factor=8) / 64
        pad_density_map_tensor = torch.zeros((1, 1, y.shape[2], y.shape[3])).cuda()
        pad_density_map_tensor[:, 0, :y_pred.shape[2], :y_pred.shape[3]] = y_pred
        y_pred = pad_density_map_tensor

        # y_max = torch.max(y)
        # y_pred_max = torch.max(y_pred)
        # max_value = y_max
        # psnr_metric = piq.psnr(y, y_pred, reduction="sum", data_range=max_value.item())
        # psnr_metric = torch.abs((y-y_pred).sum())

        # self calculate
        EPS = 1e-20
        mse = torch.mean((y_pred - y) ** 2, dim=[2, 3])
        score = - 10 * torch.log10(mse + EPS)
        psnr_metric = score

        self._sum += psnr_metric.item()
        # we multiply because ssim calculate mean of each image in batch
        # we multiply so we will divide correctly

        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CrowdCountingMeanPSNR must have at least one example before it can be computed.')
        return self._sum / self._num_examples

#################3


class CrowdCountingMeanSSIMclamp(Metric):
    """
    Calculates ssim
    require package https://github.com/photosynthesis-team/piq
    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred = output[0]
        y = output[1]
        y_pred = torch.clamp_min(y_pred, min=0.0)
        y = torch.clamp_min(y, min=0.0)
        # print("CrowdCountingMeanSSIMclamp ")
        # print("y_pred", y_pred.shape)
        # print("y", y.shape)

        y_pred = F.interpolate(y_pred, scale_factor=8) / 64
        pad_density_map_tensor = torch.zeros((1, 1, y.shape[2], y.shape[3])).cuda()
        pad_density_map_tensor[:, 0, :y_pred.shape[2], :y_pred.shape[3]] = y_pred
        y_pred = pad_density_map_tensor

        y_max = torch.max(y)
        y_pred_max = torch.max(y_pred)
        max_value = torch.max(y_max, y_pred_max)
        ssim_metric = piq.ssim(y, y_pred, reduction="sum", data_range=max_value.item())

        self._sum += ssim_metric.item()
        # we multiply because ssim calculate mean of each image in batch
        # we multiply so we will divide correctly

        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CrowdCountingMeanSSIM must have at least one example before it can be computed.')
        return self._sum / self._num_examples


class CrowdCountingMeanPSNRclamp(Metric):
    """
    Calculates ssim
    require package https://github.com/photosynthesis-team/piq
    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred = output[0]
        y_pred = torch.clamp_min(y_pred, min=0.0)
        y = output[1]
        y = torch.clamp_min(y, min=0.0)
        # print("CrowdCountingMeanPSNRclamp ")
        # print("y_pred", y_pred.shape)
        # print("y", y.shape)

        y_pred = F.interpolate(y_pred, scale_factor=8) / 64
        pad_density_map_tensor = torch.zeros((1, 1, y.shape[2], y.shape[3])).cuda()
        pad_density_map_tensor[:, 0, :y_pred.shape[2], :y_pred.shape[3]] = y_pred
        y_pred = pad_density_map_tensor

        y_max = torch.max(y)
        y_pred_max = torch.max(y_pred)
        max_value = torch.max(y_max, y_pred_max)
        psnr_metric = piq.psnr(y, y_pred, reduction="sum", data_range=max_value.item())

        self._sum += psnr_metric.item()
        # we multiply because ssim calculate mean of each image in batch
        # we multiply so we will divide correctly

        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CrowdCountingMeanPSNR must have at least one example before it can be computed.')
        return self._sum / self._num_examples
