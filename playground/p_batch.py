from data_flow import get_dataloader, create_image_list
from hard_code_variable import HardCodeVariable
import os

hard_code = HardCodeVariable()


TRAIN_PATH = os.path.join(hard_code.SHANGHAITECH_PATH_PART_B, hard_code.SHANGHAITECH_PATH_TRAIN_POSTFIX)
TEST_PATH = os.path.join(hard_code.SHANGHAITECH_PATH_PART_B, hard_code.SHANGHAITECH_PATH_TEST_POSTFIX)

train_list = create_image_list(TRAIN_PATH)
test_list = create_image_list(TEST_PATH)

train, valid, test = get_dataloader(train_list, None, test_list, dataset_name="shanghaitech", batch_size=5)

for img, label in train:
    print("img shape:" + str(img.shape) + " == " + "label shape " + str(label.shape))