import os
import glob


class HardCodeVariable():
    """
    Where did you put your data ?
    set it here, so you don't have to set in every script
    """
    def __init__(self):
        self.UCF_CC_50_PATH = "/data/cv_data/UCFCrowdCountingDataset_CVPR13_with_people_density_map/UCF_CC_50"
        self.SHANGHAITECH_PATH = "/data/ShanghaiTech"
        self.SHANGHAITECH_PATH_PART_A = "/data/ShanghaiTech/part_A/"
        self.SHANGHAITECH_PATH_PART_B = "/data/ShanghaiTech/part_B/"
        self.SHANGHAITECH_PATH_TRAIN_POSTFIX = "train_data"
        self.SHANGHAITECH_PATH_TEST_POSTFIX = "test_data"


if __name__ == "__main__":
    """
    Test if path is working 
    """
    hard_code =HardCodeVariable()
    if os.path.exists(hard_code.SHANGHAITECH_PATH):
        print("exist " + hard_code.SHANGHAITECH_PATH)
        print("let see if we have train, test folder")
        train_path_a = os.path.join(hard_code.SHANGHAITECH_PATH_PART_A, hard_code.SHANGHAITECH_PATH_TRAIN_POSTFIX)
        train_path_b = os.path.join(hard_code.SHANGHAITECH_PATH_PART_B, hard_code.SHANGHAITECH_PATH_TRAIN_POSTFIX)
        test_path_a = os.path.join(hard_code.SHANGHAITECH_PATH_PART_A, hard_code.SHANGHAITECH_PATH_TEST_POSTFIX)
        test_path_b = os.path.join(hard_code.SHANGHAITECH_PATH_PART_B, hard_code.SHANGHAITECH_PATH_TEST_POSTFIX)
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
            count_img = len(image_path_list)
            first_img = image_path_list[0]
            print("in folder " + image_root_path)
            print("--- there are total " + str(count_img))
            print('--- first img ' + first_img)