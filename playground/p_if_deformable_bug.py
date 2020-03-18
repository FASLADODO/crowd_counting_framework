from models import TorchVisionBasicDeformConv2d
import torch

if __name__ == "__main__":
    cnn = TorchVisionBasicDeformConv2d(3, 10, 3, padding=1)
    cnn2 = TorchVisionBasicDeformConv2d(3, 10, 3, padding=1)
    single_sample = torch.rand((1, 3, 24, 24))
    two_sample = torch.rand((2, 3, 24, 24))
    print(single_sample.shape)
    r_single_sample = cnn(single_sample)
    r_two_sample = cnn2(two_sample)
    print(r_two_sample.shape)