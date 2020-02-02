"""
Store data path here
"""
import os

SHANGHAITECH_PATH = "/home/tt/project/crowd-counting-data/ShanghaiTech"


class ShanghaiTechDataPath:

    def __init__(self, root=SHANGHAITECH_PATH):
        self.root = root
        self.part_a_path = os.path.join(self.root, "part_A")
        self.part_b_path = os.path.join(self.root, "part_B")

    def get(self):
        return self.root

    def get_a(self):
        return _ShanghaiTechPart(self.part_a_path)

    def get_b(self):
        return _ShanghaiTechPart(self.part_b_path)


class _ShanghaiTechPart:
    def __init__(self, root):
        self.root = root
        self.train_path = os.path.join(self.root, "train_data")
        self.test_path = os.path.join(self.root, "test_data")

    def get_train(self):
        return _ShanghaiTechData(self.train_path)

    def get_test(self):
        return _ShanghaiTechData(self.test_path)


class _ShanghaiTechData:

    def __init__(self, root):
        self.root = root
        self.images_path = os.path.join(self.root, "images")
        self.ground_truth_mat_path = os.path.join(self.root, "ground-truth")
        self.ground_truth_h5_path = os.path.join(self.root, 'ground-truth-h5')
        self.perspective_path = os.path.join(self.root, "pmap")

    def get(self):
        return self.root

    def get_images(self):
        return self.images_path

    def get_anotation_point(self):
        return self.ground_truth_mat_path

    def get_density_map(self):
        return self.ground_truth_h5_path

    def get_perspective_map(self):
        return self.perspective_path


# SHANGHAITECH_PART_A = os.path.join(SHANGHAITECH, "part_A")
# SHANGHAITECH_PART_A_train = os.path.join(SHANGHAITECH_PART_A, "part_A")
#
#
# SHANGHAITECH_PART_B = os.path.join(SHANGHAITECH, "part_B")
