
class MAECalculator:
    def __init__(self):
        self.count = 0
        self.mae_sum = 0

    def get_mae(self):
        return self.mae_sum / self.count

    def eval(self, y_pred, y_true):
        pred_count = y_pred.sum()
        true_count = y_true.sum()
        mae = abs(pred_count - true_count)
        self.mae_sum += mae
        self.count += 1

    def reset(self):
        self.count = 0
        self.mae_sum = 0