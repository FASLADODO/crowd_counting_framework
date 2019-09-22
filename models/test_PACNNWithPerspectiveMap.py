from unittest import TestCase
from models.pacnn import PACNNWithPerspectiveMap
import torch

class TestPACNNWithPerspectiveMap(TestCase):

    def test_avg_schema_pacnn(self):
        net = PACNNWithPerspectiveMap()
        # image
        # batch size, channel, h, w
        image = torch.rand(1, 3, 224, 224)
        density_map = net(image)
        print(density_map.size())

    def test_perspective_aware_schema_pacnn(self):
        net = PACNNWithPerspectiveMap(perspective_aware_mode=True)
        # image
        # batch size, channel, h, w
        image = torch.rand(1, 3, 224, 224)
        density_map = net(image)
        print(density_map.size())
