from unittest import TestCase
import os

from hard_code_variable import HardCodeVariable
from .data_path import ShanghaiTechDataPath
import glob


class TestShanghaiTechDataPath(TestCase):

    def test_get(self):
        self.test_root = "unittest_data/mock_shanghaitech"
        self.data = ShanghaiTechDataPath(root=self.test_root)
        print(self.data.get())

    def test_get_a(self):
        self.test_root = "unittest_data/mock_shanghaitech"
        self.data = ShanghaiTechDataPath(root=self.test_root)
        print(self.data.get_a())

    def test_get_b(self):
        self.test_root = "unittest_data/mock_shanghaitech"
        self.data = ShanghaiTechDataPath(root=self.test_root)
        print(self.data.get_b())

    def test_get_hc(self):
        hc = HardCodeVariable()
        self.test_root = hc.SHANGHAITECH_PATH
        self.data = ShanghaiTechDataPath(root=self.test_root)
        print(self.data.get())

    def test_get_a_hc(self):
        hc = HardCodeVariable()
        self.test_root = hc.SHANGHAITECH_PATH
        self.data = ShanghaiTechDataPath(root=self.test_root)
        print(self.data.get_a())

    def test_get_b_hc(self):
        hc = HardCodeVariable()
        self.test_root = hc.SHANGHAITECH_PATH
        self.data = ShanghaiTechDataPath(root=self.test_root)
        print(self.data.get_b())

    def testIntergration(self):
        hard_code = HardCodeVariable()
        self.test_root = hard_code.SHANGHAITECH_PATH
        self.data = ShanghaiTechDataPath(root=self.test_root)
        if os.path.exists(self.data.get()):
            print("exist " + self.data.get())
            print("let see if we have train, test folder")
            train_path_a = self.data.get_a().get_train().get()
            train_path_b = self.data.get_b().get_train().get()
            test_path_a = self.data.get_a().get_test().get()
            test_path_b = self.data.get_a().get_test().get()
            if os.path.exists(train_path_a):
                print("exist " + train_path_a)
            if os.path.exists(train_path_b):
                print("exist " + train_path_b)
            if os.path.exists(test_path_a):
                print("exist " + test_path_a)
            if os.path.exists(test_path_b):
                print("exist " + test_path_b)
            print("count number of image")
            image_folder_list = [train_path_a, train_path_b, test_path_a, test_path_b]
            for image_root_path in image_folder_list:
                image_path_list = glob.glob(os.path.join(image_root_path, "images", "*.jpg"))
                density_path_list = glob.glob(os.path.join(image_root_path, "ground-truth-h5", "*.h5"))
                count_img = len(image_path_list)
                count_density_map = len(density_path_list)
                first_img = image_path_list[0]
                first_density_map = density_path_list[0]
                print("in folder " + image_root_path)
                print("--- total image" + str(count_img))
                print('--- first img ' + first_img)
                print("--- total density map " + str(count_density_map))
                print("--- first density map " + str(first_density_map))
                if count_img == count_density_map:
                    print("--- number of density map = number of image")
                else:
                    print("xxxxx number of density map !!!!!= number of image")
                assert count_img == count_density_map
