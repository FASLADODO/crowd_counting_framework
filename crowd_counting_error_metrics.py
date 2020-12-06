from __future__ import division

import torch
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


class CrowdCountingMeanSSIM(Metric):
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
        y_pred = torch.clamp_min(y_pred, min=0)
        ssim_metric = piq.ssim(y, y_pred)

        self._sum += ssim_metric.item() * y.shape[0]
        # we multiply because ssim calculate mean of each image in batch
        # we multiply so we will divide correctly

        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CrowdCountingMeanSSIM must have at least one example before it can be computed.')
        return self._sum / self._num_examples


class CrowdCountingMeanPSNR(Metric):
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
        y_pred = torch.clamp_min(y_pred, min=0)
        y = output[1]
        psnr_metric = piq.psnr(y, y_pred)

        self._sum += psnr_metric.item() * y.shape[0]
        # we multiply because ssim calculate mean of each image in batch
        # we multiply so we will divide correctly

        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CrowdCountingMeanPSNR must have at least one example before it can be computed.')
        return self._sum / self._num_examples

