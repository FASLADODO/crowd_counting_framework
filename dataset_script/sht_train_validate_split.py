from sklearn.model_selection import train_test_split
import glob
import os
from shutil import copyfile
from args_util import train_test_split_parse


def copy_data(image_list, dest_path):
    for image_path in image_list:
        gt_path = image_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
        gt_mat_path = image_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')

        # dest
        dest_image_path = image_path.replace(DATA_PATH, dest_path)
        dest_gt_path = gt_path.replace(DATA_PATH, dest_path)
        dest_gt_mat_path = gt_mat_path.replace(DATA_PATH, dest_path)

        copyfile(image_path, dest_image_path)
        copyfile(gt_path, dest_gt_path)
        copyfile(gt_mat_path, dest_gt_mat_path)
        print("copy ", image_path, dest_image_path)


if __name__ == "__main__":

    # should contain 3 sub-folder: image, ground-truth, ground-truth-h5
    # hey, no trailing slash
    DATA_PATH = "/data/shanghaitech_with_people_density_map/ShanghaiTech/part_A/train_data"

    args = train_test_split_parse()
    DATA_PATH = args.input

    # get list of sample
    image_list = glob.glob(os.path.join(DATA_PATH, "images", "*.jpg"))

    train_image_list, validate_image_list  =  train_test_split(image_list, test_size=0.2, random_state=19051890)

    print("train count ", len(train_image_list))
    print("validate count ", len(validate_image_list))


    # remove last slash
    if DATA_PATH[-1] == "/":
        DATA_PATH = DATA_PATH[:-1]

    # make train, validate path
    train_path = DATA_PATH + "_train_split"
    validate_path = DATA_PATH + "_validate_split"

    # make dir
    os.makedirs(os.path.join(train_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_path, "ground-truth-h5"), exist_ok=True)
    os.makedirs(os.path.join(train_path, "ground-truth"), exist_ok=True)

    os.makedirs(os.path.join(validate_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(validate_path, "ground-truth-h5"), exist_ok=True)
    os.makedirs(os.path.join(validate_path, "ground-truth"), exist_ok=True)

    copy_data(train_image_list, train_path)
    copy_data(validate_image_list, validate_path)

