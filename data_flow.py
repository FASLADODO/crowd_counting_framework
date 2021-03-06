import os
import glob
from sklearn.model_selection import train_test_split
import json
import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
import scipy.io  # import scipy does not work https://stackoverflow.com/questions/11172623/import-problems-with-scipy-io
from data_util.dataset_utils import my_collate, flatten_collate

"""
create a list of file (full directory)
"""


def count_gt_annotation_sha(mat_path):
    """
    read the annotation and count number of head from annotation
    :param mat_path:
    :return: count
    """
    mat = scipy.io.loadmat(mat_path, appendmat=False)
    gt = mat["image_info"][0, 0][0, 0][0]
    return len(gt)

def count_gt_annotation_jhu(txt_path):
    """
    read the annotation and count number of head from annotation
    :param mat_path:
    :return: count
    """
    result = 0
    try:
        df = pd.read_csv(txt_path, sep=" ", header=None)
        p = df.to_numpy()
        result = len(p)
    except Exception:
        print("exception count_gt_annotation_jhu ", txt_path)
    return result



def create_training_image_list(data_path):
    """
    create a list of absolutely path of jpg file
    :param data_path: must contain subfolder "images" with *.jpg  (example ShanghaiTech/part_A/train_data/)
    :return:
    """
    DATA_PATH = data_path
    image_path_list = glob.glob(os.path.join(DATA_PATH, "images", "*.jpg"))
    return image_path_list


def create_image_list(data_path):
    DATA_PATH = data_path
    image_path_list = glob.glob(os.path.join(DATA_PATH, "images", "*.jpg"))
    return image_path_list


def get_train_val_list(data_path, test_size=0.1):
    DATA_PATH = data_path
    image_path_list = glob.glob(os.path.join(DATA_PATH, "images", "*.jpg"))
    if len(image_path_list) is 0:
        image_path_list = glob.glob(os.path.join(DATA_PATH, "*.jpg"))
    train, val = train_test_split(image_path_list, test_size=test_size)

    print("train size ", len(train))
    print("val size ", len(val))
    return train, val


def load_data(img_path, train=True):
    """
    get a sample
    :deprecate: use load_data_shanghaiTech now
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    gt_file.close()

    target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),
                        interpolation=cv2.INTER_CUBIC) * 64

    return img, target


def load_data_shanghaitech(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    gt_file.close()

    if train:
        crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
        if random.randint(0, 9) <= -1:

            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),
                         interpolation=cv2.INTER_CUBIC) * 64
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    return img, target1


def load_data_shanghaitech_rnd(img_path, train=True):
    """
    crop 1/4 image, but random
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    gt_file.close()
    if train:
        crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
        if random.randint(0, 9) <= 4:
            # crop 4 corner
            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            # crop random
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),
                         interpolation=cv2.INTER_CUBIC) * 64
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        return img, gt_count

    return img, target1


def load_data_shanghaitech_more_rnd(img_path, train=True):
    """
    crop 1/4 image, but random
    increase random crop chance (reduce 1/4 four corner chance)
    increase flip chance to 50%
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])

    if train:
        crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
        if random.randint(0, 9) <= 3:
            # crop 4 corner
            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            # crop random
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.5:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        return img, gt_count

    target1 = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),
                         interpolation=cv2.INTER_CUBIC) * 64
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    return img, target1


def load_data_shanghaitech_20p_enlarge(img_path, train=True):
    """
    20 percent crop, then enlarge to equal size of original
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        if random.random() > 0.8:
            crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
            if random.randint(0, 9) <= -1:

                dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
                dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
            else:
                dx = int(random.random() * img.size[0] * 1. / 2)
                dy = int(random.random() * img.size[1] * 1. / 2)

            img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
            target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

            # enlarge image patch to original size
            img = img.resize((crop_size[0] * 2, crop_size[1] * 2), Image.ANTIALIAS)
            target_factor = 4  # thus, target is not enlarge, so output target only / 4

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    return img, target1


def load_data_shanghaitech_20p(img_path, train=True):
    """
    20 percent crop
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        if random.random() > 0.8:
            crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
            if random.randint(0, 9) <= -1:

                dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
                dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
            else:
                dx = int(random.random() * img.size[0] * 1. / 2)
                dy = int(random.random() * img.size[1] * 1. / 2)

            img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
            target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

            # # enlarge image patch to original size
            # img = img.resize((crop_size[0]*2, crop_size[1]*2), Image.ANTIALIAS)
            # target_factor = 4 # thus, target is not enlarge, so output target only / 4

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        return img, gt_count

    return img, target1


def load_data_shanghaitech_40p(img_path, train=True):
    """
    20 percent crop
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        if random.random() > 0.6:
            crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
            if random.randint(0, 9) <= -1:

                dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
                dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
            else:
                dx = int(random.random() * img.size[0] * 1. / 2)
                dy = int(random.random() * img.size[1] * 1. / 2)

            img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
            target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

            # # enlarge image patch to original size
            # img = img.resize((crop_size[0]*2, crop_size[1]*2), Image.ANTIALIAS)
            # target_factor = 4 # thus, target is not enlarge, so output target only / 4

        if random.random() > 0.6:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    return img, target1


def load_data_shanghaitech_20p_random(img_path, train=True):
    """
    20 percent crop
    now it is also random crop, not just crop 4 corner
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        if random.random() > 0.8:
            crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
            if random.randint(0, 9) <= 3:  # crop 4 corner, 40% chance
                dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
                dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
            else:  # crop random, 60% chance
                dx = int(random.random() * img.size[0] * 1. / 2)
                dy = int(random.random() * img.size[1] * 1. / 2)

            img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
            target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

            # # enlarge image patch to original size
            # img = img.resize((crop_size[0]*2, crop_size[1]*2), Image.ANTIALIAS)
            # target_factor = 4 # thus, target is not enlarge, so output target only / 4

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        return img, gt_count

    return img, target1


def load_data_shanghaitech_60p_random(img_path, train=True):
    """
    40 percent crop
    now it is also random crop, not just crop 4 corner
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        if random.random() > 0.4:
            crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
            if random.randint(0, 9) <= 3:  # crop 4 corner, 40% chance
                dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
                dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
            else:  # crop random, 60% chance
                dx = int(random.random() * img.size[0] * 1. / 2)
                dy = int(random.random() * img.size[1] * 1. / 2)

            img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
            target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

            # # enlarge image patch to original size
            # img = img.resize((crop_size[0]*2, crop_size[1]*2), Image.ANTIALIAS)
            # target_factor = 4 # thus, target is not enlarge, so output target only / 4

        if random.random() > 0.5:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        return img, gt_count

    return img, target1


def load_data_shanghaitech_flip_only(img_path, train=True):
    """
    flip only
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img_origin = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target_origin = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        # for each image
        # make 2, original and flip
        crop_img = []
        crop_label = []
        # flip
        for x in range(2):
            if x == 1:
                target = np.fliplr(target_origin)
                img = img_origin.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                target = target_origin
                img = img_origin
            target1 = cv2.resize(target,
                                 (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                                 interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
            # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
            target1 = np.expand_dims(target1,
                                     axis=0)  # make dim (batch size, channel size, x, y) to make model output
            crop_img.append(img)
            crop_label.append(target1)

        return crop_img, crop_label

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        return img_origin, gt_count


def load_data_my_bike_non_overlap(img_path, train=True, debug=False):
    """
    per sample, crop 4, non-overlap
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img_origin = Image.open(img_path).convert('RGB')
    crop_size = (int(img_origin.size[0] / 2), int(img_origin.size[1] / 2))
    gt_file = h5py.File(gt_path, 'r')
    target_origin = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        # for each image
        # create 8 patches, 4 non-overlap 4 corner
        # for each of 4 patch, create another 4 flip
        crop_img = []
        crop_label = []
        for i in range(2):
            for j in range(2):
                # crop non-overlap
                dx = int(i * img_origin.size[0] * 1. / 2)
                dy = int(j * img_origin.size[1] * 1. / 2)
                img = img_origin.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
                target = target_origin[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

                # flip
                for x in range(2):
                    if x == 1:
                        target = np.fliplr(target)
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    target1 = cv2.resize(target,
                                         (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
                    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
                    target1 = np.expand_dims(target1,
                                             axis=0)  # make dim (batch size, channel size, x, y) to make model output
                    crop_img.append(img)
                    crop_label.append(target1)

        return crop_img, crop_label

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.json').replace('images', 'jsons')
        gt_count = get_my_bike_count_from_json(mat_path)
        if debug:
            gt_file = h5py.File(gt_path, 'r')
            target = np.asarray(gt_file['density'])
            return img_origin, gt_count, target
        return img_origin, gt_count


def load_data_shanghaitech_non_overlap(img_path, train=True, debug=False):
    """
    per sample, crop 4, non-overlap
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img_origin = Image.open(img_path).convert('RGB')
    crop_size = (int(img_origin.size[0] / 2), int(img_origin.size[1] / 2))
    gt_file = h5py.File(gt_path, 'r')
    target_origin = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        # for each image
        # create 8 patches, 4 non-overlap 4 corner
        # for each of 4 patch, create another 4 flip
        crop_img = []
        crop_label = []
        for i in range(2):
            for j in range(2):
                # crop non-overlap
                dx = int(i * img_origin.size[0] * 1. / 2)
                dy = int(j * img_origin.size[1] * 1. / 2)
                img = img_origin.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
                target = target_origin[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

                # flip
                for x in range(2):
                    if x == 1:
                        target = np.fliplr(target)
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    target1 = cv2.resize(target,
                                         (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
                    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
                    target1 = np.expand_dims(target1,
                                             axis=0)  # make dim (batch size, channel size, x, y) to make model output
                    crop_img.append(img)
                    crop_label.append(target1)

        return crop_img, crop_label

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        if debug:
            gt_file = h5py.File(gt_path, 'r')
            target = np.asarray(gt_file['density'])
            return img_origin, gt_count, target
        return img_origin, gt_count

def load_data_shanghaitech_non_overlap_test_with_densitygt(img_path, train=True, debug=False):
    """
    per sample, crop 4, non-overlap
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img_origin = Image.open(img_path).convert('RGB')
    crop_size = (int(img_origin.size[0] / 2), int(img_origin.size[1] / 2))
    gt_file = h5py.File(gt_path, 'r')
    target_origin = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        # for each image
        # create 8 patches, 4 non-overlap 4 corner
        # for each of 4 patch, create another 4 flip
        crop_img = []
        crop_label = []
        for i in range(2):
            for j in range(2):
                # crop non-overlap
                dx = int(i * img_origin.size[0] * 1. / 2)
                dy = int(j * img_origin.size[1] * 1. / 2)
                img = img_origin.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
                target = target_origin[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

                # flip
                for x in range(2):
                    if x == 1:
                        target = np.fliplr(target)
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    target1 = cv2.resize(target,
                                         (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
                    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
                    target1 = np.expand_dims(target1,
                                             axis=0)  # make dim (batch size, channel size, x, y) to make model output
                    crop_img.append(img)
                    crop_label.append(target1)

        return crop_img, crop_label

    if not train:
        gt_file = h5py.File(gt_path, 'r')
        target = np.asarray(gt_file['density'])
        # target1 = cv2.resize(target,
        #                      (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
        #                      interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
        # # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
        target1 = np.expand_dims(target,
                                 axis=0)  # make dim (batch size, channel size, x, y) to make model output
        return img_origin, target1


def load_data_shanghaitech_non_overlap_downsample(img_path, train=True):
    """
    input image downsample by half
    per sample, crop 4, non-overlap
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img_origin = Image.open(img_path).convert('RGB')
    # downsample by half
    img_origin = img_origin.resize((int(img_origin.size[0] / 2), int(img_origin.size[1] / 2)), resample=Image.ANTIALIAS)
    crop_size = (int(img_origin.size[0] / 2), int(img_origin.size[1] / 2))
    gt_file = h5py.File(gt_path, 'r')
    target_origin = np.asarray(gt_file['density'])
    target_factor = 4

    if train:
        # for each image
        # create 8 patches, 4 non-overlap 4 corner
        # for each of 4 patch, create another 4 flip
        crop_img = []
        crop_label = []
        for i in range(2):
            for j in range(2):
                # crop non-overlap
                dx = int(i * img_origin.size[0] * 1. / 2)
                dy = int(j * img_origin.size[1] * 1. / 2)
                img = img_origin.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
                target = target_origin[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

                # flip
                for x in range(2):
                    if x == 1:
                        target = np.fliplr(target)
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    target1 = cv2.resize(target,
                                         (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
                    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
                    target1 = np.expand_dims(target1,
                                             axis=0)  # make dim (batch size, channel size, x, y) to make model output
                    crop_img.append(img)
                    crop_label.append(target1)

        return crop_img, crop_label

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        return img_origin, gt_count

def load_data_shanghaitech_non_overlap_noflip(img_path, train=True):
    """
    per sample, crop 4, non-overlap
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img_origin = Image.open(img_path).convert('RGB')
    crop_size = (int(img_origin.size[0] / 2), int(img_origin.size[1] / 2))
    gt_file = h5py.File(gt_path, 'r')
    target_origin = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        # for each image
        # create 8 patches, 4 non-overlap 4 corner
        # for each of 4 patch, create another 4 flip
        crop_img = []
        crop_label = []
        for i in range(2):
            for j in range(2):
                # crop non-overlap
                dx = int(i * img_origin.size[0] * 1. / 2)
                dy = int(j * img_origin.size[1] * 1. / 2)
                img = img_origin.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
                target = target_origin[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

                target1 = cv2.resize(target,
                                     (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                                     interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
                target1 = np.expand_dims(target1,
                                         axis=0)  # make dim (batch size, channel size, x, y) to make model output
                crop_img.append(img)
                crop_label.append(target1)
        return crop_img, crop_label

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        return img_origin, gt_count


def load_data_shanghaitech_crop_random(img_path, train=True):
    """
    40 percent crop
    now it is also random crop, not just crop 4 corner
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
        if random.randint(0, 9) <= 1:  # crop 4 corner, 20% chance
            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:  # crop random, 80% chance
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        # # enlarge image patch to original size
        # img = img.resize((crop_size[0]*2, crop_size[1]*2), Image.ANTIALIAS)
        # target_factor = 4 # thus, target is not enlarge, so output target only / 4

        if random.random() > 0.5:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        return img, gt_count

    return img, target1


def load_data_shanghaitech_180(img_path, train=True):
    """
    crop fixed 180, allow batch in non-uniform dataset
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    target_factor = 8

    if train:
        crop_size = (180, 180)
        dx = int(random.random() * (img.size[0] - 180))
        dy = int(random.random() * (img.size[1] - 180))
        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    return img, target1


def load_data_shanghaitech_256(img_path, train=True):
    """
    crop fixed 256, allow batch in non-uniform dataset
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    gt_file.close()
    target_factor = 8
    crop_sq_size = 256
    if train:
        crop_size = (crop_sq_size, crop_sq_size)
        dx = int(random.random() * (img.size[0] - crop_sq_size))
        dy = int(random.random() * (img.size[1] - crop_sq_size))
        if img.size[0] - crop_sq_size < 0 or img.size[1] - crop_sq_size < 0:  # we crop more than we can chew, so...
            return None, None
        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    return img, target1

def load_data_shanghaitech_256_v2(img_path, train=True):
    """
    crop fixed 256, allow batch in non-uniform dataset
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img_origin = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    gt_file.close()
    target_factor = 8
    crop_sq_size = 256
    if train:
        crop_size = (crop_sq_size, crop_sq_size)
        dx = int(random.random() * (img_origin.size[0] - crop_sq_size))
        dy = int(random.random() * (img_origin.size[1] - crop_sq_size))
        if img_origin.size[0] - crop_sq_size < 0 or img_origin.size[1] - crop_sq_size < 0:  # we crop more than we can chew, so...
            # TODO if exception, do somehthing here
            return None, None
        img = img_origin.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        return img_origin, gt_count

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    return img, target1

def load_data_shanghaitech_same_size_density_map(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])

    if train:
        crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
        if random.randint(0, 9) <= -1:

            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = target
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    return img, target1


def load_data_shanghaitech_keepfull(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])

    if train:
        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),
                         interpolation=cv2.INTER_CUBIC) * 64
    if not train:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        return img, gt_count

    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    # np.expand_dims(target1, axis=0)  # again
    return img, target1


def load_data_shanghaitech_keepfull_r50(img_path, train=True, debug=False):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')

    if train:
        gt_file = h5py.File(gt_path, 'r')
        target = np.asarray(gt_file['density'])
        if random.random() > 0.5:  # flip o.5 chance
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        target1 = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),
                             interpolation=cv2.INTER_CUBIC) * 64
        target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
        # np.expand_dims(target1, axis=0)  # again
        return img, target1
    else:
        # get correct people head count from head annotation
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG', 'GT_IMG')
        gt_count = count_gt_annotation_sha(mat_path)
        if debug:
            gt_file = h5py.File(gt_path, 'r')
            target = np.asarray(gt_file['density'])
            return img, gt_count, target
        return img, gt_count

def load_data_shanghaitech_keepfull_and_crop(img_path, train=True):
    """
    loader might give full image, or crop
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])

    if train:

        if random.random() > 0.5:  # 50% chance crop
            crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
            if random.randint(0, 9) <= -1:

                dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
                dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
            else:
                dx = int(random.random() * img.size[0] * 1. / 2)
                dy = int(random.random() * img.size[1] * 1. / 2)

            img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
            target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:  # 20 % chance flip
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),
                         interpolation=cv2.INTER_CUBIC) * 64

    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    # np.expand_dims(target1, axis=0)  # again
    return img, target1


def load_data_ucf_cc50(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])

    if train:
        img, target = data_augmentation(img, target)

    target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),
                        interpolation=cv2.INTER_CUBIC) * 64

    return img, target


def load_data_shanghaitech_pacnn(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])

    if train:
        crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
        if random.randint(0, 9) <= -1:

            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),
                         interpolation=cv2.INTER_CUBIC) * 64
    target2 = cv2.resize(target, (int(target.shape[1] / 16), int(target.shape[0] / 16)),
                         interpolation=cv2.INTER_CUBIC) * 256
    target3 = cv2.resize(target, (int(target.shape[1] / 32), int(target.shape[0] / 32)),
                         interpolation=cv2.INTER_CUBIC) * 1024

    return img, (target1, target2, target3)


def load_data_shanghaitech_pacnn_with_perspective(img_path, train=True):
    """
    # TODO: TEST this
    :param img_path: should contain sub folder images (contain IMG_num.jpg), ground-truth-h5
    :param perspective_path: should contain IMG_num.mat
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    p_path = img_path.replace(".jpg", ".mat").replace("images", "pmap")
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    perspective = np.array(h5py.File(p_path, "r")['pmap']).astype(np.float32)
    perspective = np.rot90(perspective, k=3)
    if train:
        crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
        if random.randint(0, 9) <= -1:

            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
        perspective = perspective[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            perspective = np.fliplr(perspective)

    perspective /= np.max(perspective)

    target1 = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),
                         interpolation=cv2.INTER_CUBIC) * 64
    target2 = cv2.resize(target, (int(target.shape[1] / 16), int(target.shape[0] / 16)),
                         interpolation=cv2.INTER_CUBIC) * 256
    target3 = cv2.resize(target, (int(target.shape[1] / 32), int(target.shape[0] / 32)),
                         interpolation=cv2.INTER_CUBIC) * 1024

    perspective_s = cv2.resize(perspective, (int(perspective.shape[1] / 16), int(perspective.shape[0] / 16)),
                               interpolation=cv2.INTER_CUBIC)

    perspective_p = cv2.resize(perspective, (int(perspective.shape[1] / 8), int(perspective.shape[0] / 8)),
                               interpolation=cv2.INTER_CUBIC)

    return img, (target1, target2, target3, perspective_s, perspective_p)


def load_data_ucf_cc50_pacnn(img_path, train=True):
    """
    dataloader for UCF-CC-50 dataset
    label with 3 density map d1, d2, d3 for pacnn
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])

    if train:
        crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
        if random.randint(0, 9) <= -1:

            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target1 = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),
                         interpolation=cv2.INTER_CUBIC) * 64
    target2 = cv2.resize(target, (int(target.shape[1] / 16), int(target.shape[0] / 16)),
                         interpolation=cv2.INTER_CUBIC) * 256
    target3 = cv2.resize(target, (int(target.shape[1] / 32), int(target.shape[0] / 32)),
                         interpolation=cv2.INTER_CUBIC) * 1024

    return img, (target1, target2, target3)


def load_data_jhucrowd_256(img_path, train=True, debug=False):
    """
    for jhucrowd
    crop fixed 256, allow batch in non-uniform dataset
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img_origin = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density']).astype('float32')
    target_factor = 8
    crop_sq_size = 256
    if train:
        crop_size = (crop_sq_size, crop_sq_size)
        dx = int(random.random() * (img_origin.size[0] - crop_sq_size))
        dy = int(random.random() * (img_origin.size[1] - crop_sq_size))
        if img_origin.size[0] - crop_sq_size < 0 or img_origin.size[1] - crop_sq_size < 0:  # we crop more than we can chew, so...
            return None, None
        img = img_origin.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if not train:
        # get correct people head count from head annotation
        txt_path = img_path.replace('.jpg', '.txt').replace('images', 'ground-truth')
        gt_count = count_gt_annotation_jhu(txt_path)
        if debug:
            gt_file = h5py.File(gt_path, 'r')
            target = np.asarray(gt_file['density'])
            return img_origin, gt_count, target
        return img_origin, gt_count

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    return img, target1


def load_data_jhucrowd_downsample_512(img_path, train=True, debug=False):
    """
    for jhucrowd
    crop fixed 256, allow batch in non-uniform dataset
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img_origin = Image.open(img_path).convert('RGB')

    downsample_rate = 2
    target_factor = 8 * downsample_rate
    crop_sq_size = 512

    if train:
        gt_file = h5py.File(gt_path, 'r')
        target = np.asarray(gt_file['density']).astype('float32')
        crop_size = (crop_sq_size, crop_sq_size)
        dx = int(random.random() * (img_origin.size[0] - crop_sq_size))
        dy = int(random.random() * (img_origin.size[1] - crop_sq_size))
        if img_origin.size[0] - crop_sq_size < 0 or img_origin.size[1] - crop_sq_size < 0:  # we crop more than we can chew, so...
            return None, None
        img = img_origin.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        img2 = img.resize((int(img.size[0] / downsample_rate), int(img.size[1] / downsample_rate)),
                                       resample=Image.ANTIALIAS)
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)

    if not train:
        # get correct people head count from head annotation
        txt_path = img_path.replace('.jpg', '.txt').replace('images', 'ground-truth')
        gt_count = count_gt_annotation_jhu(txt_path)
        img_eval = img_origin.resize((int(img_origin.size[0] / downsample_rate), int(img_origin.size[1] / downsample_rate)),
                   resample=Image.ANTIALIAS)
        if debug:
            gt_file = h5py.File(gt_path, 'r')
            target = np.asarray(gt_file['density'])
            return img_eval, gt_count, target
        return img_eval, gt_count

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    return img2, target1



def load_data_jhucrowd_downsample_testonly_512(img_path, train=True, debug=False):
    """
    for jhucrowd
    crop fixed 256, allow batch in non-uniform dataset
    :param img_path:
    :param train:
    :return:
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')
    img_origin = Image.open(img_path).convert('RGB')

    downsample_rate = 2
    target_factor = 8
    crop_sq_size = 512

    if train:
        gt_file = h5py.File(gt_path, 'r')
        target = np.asarray(gt_file['density']).astype('float32')
        crop_size = (crop_sq_size, crop_sq_size)
        dx = int(random.random() * (img_origin.size[0] - crop_sq_size))
        dy = int(random.random() * (img_origin.size[1] - crop_sq_size))
        if img_origin.size[0] - crop_sq_size < 0 or img_origin.size[1] - crop_sq_size < 0:  # we crop more than we can chew, so...
            return None, None
        img = img_origin.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))

        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if not train:
        # get correct people head count from head annotation
        txt_path = img_path.replace('.jpg', '.txt').replace('images', 'ground-truth')
        gt_count = count_gt_annotation_jhu(txt_path)
        img_eval = img_origin.resize((int(img_origin.size[0] / downsample_rate), int(img_origin.size[1] / downsample_rate)),
                   resample=Image.ANTIALIAS)
        if debug:
            gt_file = h5py.File(gt_path, 'r')
            target = np.asarray(gt_file['density'])
            return img_eval, gt_count, target
        return img_eval, gt_count

    target1 = cv2.resize(target, (int(target.shape[1] / target_factor), int(target.shape[0] / target_factor)),
                         interpolation=cv2.INTER_CUBIC) * target_factor * target_factor
    # target1 = target1.unsqueeze(0)  # make dim (batch size, channel size, x, y) to make model output
    target1 = np.expand_dims(target1, axis=0)  # make dim (batch size, channel size, x, y) to make model output
    return img, target1

def data_augmentation(img, target):
    """
    return 1 pair of img, target after apply augmentation
    :param img:
    :param target:
    :return:
    """
    crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
    if random.randint(0, 9) <= -1:

        dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
        dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
    else:
        dx = int(random.random() * img.size[0] * 1. / 2)
        dy = int(random.random() * img.size[1] * 1. / 2)

    img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
    target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

    if random.random() > 0.8:
        target = np.fliplr(target)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img, target



class ListDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 debug=False,
                 num_workers=0, dataset_name="shanghaitech", cache=False):
        """
        if you have different image size, then batch_size must be 1
        :param root:
        :param shape:
        :param shuffle:
        :param transform:
        :param train:
        :param debug: will print path of image
        :param seen:
        :param batch_size:
        :param num_workers:
        """
        if train:
            if "non_overlap" in dataset_name or "shanghaitech_keepfull" in dataset_name:
                # each sample we generate 8 image, so, no need to x4
                print("root * 1 because dataset ", dataset_name)
            else:
                root = root * 4
                print("root * 4 because dataset ", dataset_name)
        if shuffle:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.cache = cache
        self.cache_train = {}
        self.cache_eval = {}
        self.debug = debug
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        print("in ListDataset dataset_name is |" + dataset_name + "|")
        # load data fn
        if dataset_name == "shanghaitech":
            self.load_data_fn = load_data_shanghaitech
        elif dataset_name == "shanghaitech_random":
            self.load_data_fn = load_data_shanghaitech_rnd
        if dataset_name == "shanghaitech_more_random":
            self.load_data_fn = load_data_shanghaitech_more_rnd
        elif dataset_name == "shanghaitech_same_size_density_map":
            self.load_data_fn = load_data_shanghaitech_same_size_density_map
        elif dataset_name == "shanghaitech_keepfull":
            self.load_data_fn = load_data_shanghaitech_keepfull
        elif dataset_name == "shanghaitech_keepfull_r50":
            self.load_data_fn = load_data_shanghaitech_keepfull_r50
        elif dataset_name == "shanghaitech_keepfull_and_crop":
            self.load_data_fn = load_data_shanghaitech_keepfull_and_crop
        elif dataset_name == "shanghaitech_20p_enlarge":
            self.load_data_fn = load_data_shanghaitech_20p_enlarge
        elif dataset_name == "shanghaitech_20p":
            self.load_data_fn = load_data_shanghaitech_20p
        elif dataset_name == "shanghaitech_40p":
            self.load_data_fn = load_data_shanghaitech_40p
        elif dataset_name == "shanghaitech_20p_random":
            self.load_data_fn = load_data_shanghaitech_20p_random
        elif dataset_name == "shanghaitech_60p_random":
            self.load_data_fn = load_data_shanghaitech_60p_random
        elif dataset_name == "shanghaitech_crop_random":
            self.load_data_fn = load_data_shanghaitech_crop_random
        elif dataset_name == "shanghaitech_180":
            self.load_data_fn = load_data_shanghaitech_180
        elif dataset_name == "shanghaitech_256":
            self.load_data_fn = load_data_shanghaitech_256
        elif dataset_name == "shanghaitech_256_v2":
            self.load_data_fn = load_data_shanghaitech_256_v2
        elif dataset_name == "jhucrowd_downsample_512":
            self.load_data_fn = load_data_jhucrowd_downsample_512
        elif dataset_name == "jhucrowd_downsample_testonly_512":
            self.load_data_fn = load_data_jhucrowd_downsample_testonly_512
        elif dataset_name == "shanghaitech_non_overlap":
            self.load_data_fn = load_data_shanghaitech_non_overlap
        elif dataset_name == "shanghaitech_non_overlap_test_with_densitygt":
            self.load_data_fn = load_data_shanghaitech_non_overlap_test_with_densitygt
        elif dataset_name == "shanghaitech_non_overlap_downsample":
            self.load_data_fn = load_data_shanghaitech_non_overlap_downsample
        elif dataset_name == "shanghaitech_flip_only":
            self.load_data_fn = load_data_shanghaitech_flip_only
        elif dataset_name == "jhucrowd_256":
            self.load_data_fn = load_data_jhucrowd_256
        elif dataset_name == "ucf_cc_50":
            self.load_data_fn = load_data_ucf_cc50
        elif dataset_name == "ucf_cc_50_pacnn":
            self.load_data_fn = load_data_ucf_cc50_pacnn
        elif dataset_name == "shanghaitech_pacnn":
            self.load_data_fn = load_data_shanghaitech_pacnn
        elif dataset_name == "my_bike_non_overlap":
            self.load_data_fn = load_data_my_bike_non_overlap
        elif dataset_name == "shanghaitech_pacnn_with_perspective":
            self.load_data_fn = load_data_shanghaitech_pacnn_with_perspective

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]

        # hard code to calculate ssim and psnr
        # fail dont do this
        # if self.dataset_name == "shanghaitech_non_overlap_test_with_densitygt" and not self.train:
        #     img, count, target = self.load_data_fn(img_path, self.train)
        #     if self.transform is not None:
        #         if isinstance(img, list):
        #             # for case of generate  multiple augmentation per sample
        #             img_r = [self.transform(img_item) for img_item in img]
        #             img = img_r
        #         else:
        #             img = self.transform(img)
        #     return img, count, target

        # if self.debug:
        #     print(img_path)
        # try to check cache item if exist
        if self.cache and self.train and index in self.cache_train.keys():
            img, target = self.cache_train[index]
        elif self.cache and not self.train and index in self.cache_eval.keys():
            img, target = self.cache_eval[index]
        # no cache, load data as usual
        else:
            img, target = self.load_data_fn(img_path, self.train)
            if img is None or target is None:
                return None
            if self.transform is not None:
                if isinstance(img, list):
                    # for case of generate  multiple augmentation per sample
                    img_r = [self.transform(img_item) for img_item in img]
                    img = img_r
                else:
                    img = self.transform(img)
            # if use cache, then save data to cache
            if self.cache:
                if self.train:
                    self.cache_train[index] = (img, target)
                else:
                    self.cache_eval[index] = (img, target)
        if self.debug:
            if self.train:
                _, p_count = self.load_data_fn(img_path, train=False)
                # print(img_path + " " + str(target.sum()) + " " + str(p_count))
                img_name = img_path.split("/")[-1]
                # when debug, give information on p_count and img_name
                debug_info = {"p_count": p_count,
                              "name": img_name}
                return img, target, debug_info
            else:  # train=False
                # when dataloader is not in train mode
                # we get density gt, p_count from annotation
                _, p_count, target = self.load_data_fn(img_path, train=False, debug=True)
                print(img_path + " " + str(target.sum()) + " " + str(p_count))
                img_name = img_path.split("/")[-1]
                # when debug, give information on p_count and img_name
                debug_info = {"p_count": p_count,
                              "name": img_name}
                return img, target, debug_info
        else:
            return img, target


def get_dataloader(train_list, val_list, test_list, dataset_name="shanghaitech", visualize_mode=False, batch_size=1,
                   train_loader_for_eval_check=False, cache=False, pin_memory=False,
                   debug=False, test_size=1):

    if visualize_mode:
        transformer = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]),
        ])
    train_collate_fn = my_collate
    if dataset_name == "shanghaitech_non_overlap" or\
        dataset_name == "shanghaitech_non_overlap_test_with_densitygt" or \
        dataset_name == "shanghaitech_non_overlap_downsample" or \
        dataset_name == "my_bike_non_overlap" or \
        dataset_name == "shanghaitech_flip_only":
        train_collate_fn = flatten_collate
    train_loader = torch.utils.data.DataLoader(
        ListDataset(train_list,
                    shuffle=True,
                    transform=transformer,
                    train=True,
                    batch_size=batch_size,
                    num_workers=0,
                    debug=debug,
                    dataset_name=dataset_name, cache=cache),
        batch_size=batch_size,
        num_workers=0,
        collate_fn=train_collate_fn, pin_memory=pin_memory)

    train_loader_for_eval = torch.utils.data.DataLoader(
        ListDataset(train_list,
                    shuffle=False,
                    transform=transformer,
                    train=False,
                    batch_size=batch_size,
                    num_workers=0,
                    debug=debug,
                    dataset_name=dataset_name, cache=cache),
        batch_size=1,
        num_workers=0,
        collate_fn=my_collate, pin_memory=pin_memory)

    if val_list is not None:
        val_loader = torch.utils.data.DataLoader(
            ListDataset(val_list,
                        shuffle=False,
                        transform=transformer,
                        train=False,
                        debug=debug,
                        batch_size=test_size,
                        dataset_name=dataset_name, cache=True),  # evaluation set always cache
            num_workers=0,
            batch_size=test_size,
            pin_memory=pin_memory)
    else:
        val_loader = None

    if test_list is not None:
        test_loader = torch.utils.data.DataLoader(
            ListDataset(test_list,
                        shuffle=False,
                        transform=transformer,
                        train=False,
                        debug=debug,
                        batch_size=test_size,
                        dataset_name=dataset_name, cache=True),  # evaluation set always cache
            num_workers=0,
            batch_size=test_size,
            pin_memory=pin_memory)
    else:
        test_loader = None
    if train_loader_for_eval_check:
        return train_loader, train_loader_for_eval, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader


def simple_predict_data_load_fn(img_path):
    img_name = img_path.split("/")[-1]
    # when debug, give information on p_count and img_name
    img_id = img_name.split(".")[0].split("_")[-1]
    debug_info = {"img_path":img_path,
                  "name": img_name,
                  "id": img_id}
    img_origin = Image.open(img_path).convert('RGB')
    return img_origin, debug_info


class PredictListDataset(Dataset):
    """
    only to predict without label
    """
    def __init__(self, root, shape=None, shuffle=True, transform=None, batch_size=1,
                 debug=False,
                 num_workers=0):
        """
        if you have different image size, then batch_size must be 1
        :param root:
        :param shape:
        :param shuffle:
        :param transform:
        :param train:
        :param debug: will print path of image
        :param seen:
        :param batch_size:
        :param num_workers:
        """

        if shuffle:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform

        self.cache_train = {}
        self.cache_eval = {}
        self.debug = debug
        self.shape = shape

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        # if self.debug:
        #     print(img_path)
        # try to check cache item if exist
        img, info = simple_predict_data_load_fn(img_path)
        if self.transform is not None:
            if isinstance(img, list):
                # for case of generate  multiple augmentation per sample
                img_r = [self.transform(img_item) for img_item in img]
                img = img_r
            else:
                img = self.transform(img)
        return img, info


from torchvision.io.video import read_video, write_video


class PredictVideoDataset(Dataset):
    """
    only to predict without label
    """
    def __init__(self, video_path, shape=None, transform=None, batch_size=1,
                 debug=False,
                 num_workers=0):
        """
        if you have different image size, then batch_size must be 1
        :param video_path: path to video file
        :param shape:
        :param shuffle:
        :param transform:
        :param debug: will print path of image
        :param batch_size:
        :param num_workers:
        """
        v, a, info = read_video(video_path, pts_unit='sec')

        self.video_tensor = v
        self.nSamples = self.video_tensor.shape[0]

        self.transform = transform
        self.debug = debug
        self.shape = shape

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        info = {"index": index}
        img = self.video_tensor[index]
        img = img.permute(2, 0, 1)
        img = img.float()
        if self.transform is not None:
            img = self.transform(img)
        return img, info

def get_predict_video_dataloader(video_path, visualize_mode = False, batch_size = 1):
    if visualize_mode:
        transformer = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]),
        ])

    # if not visualize_mode:
    #     transformer = transforms.Compose([
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225]),
    #     ])
    # else:
    #     transformer = None

    loader = torch.utils.data.DataLoader(PredictVideoDataset(
        video_path,
        batch_size= batch_size,
        transform= transformer
    ),
        batch_size= batch_size,
    )

    return loader


def get_my_bike_count_from_json(json_full_path):
    count_human = 0
    with open(json_full_path) as json_file:
        json_data = json.load(json_file)
        count_human = json_data['human_num']
    return count_human


def get_predict_dataloader(data_list, visualize_mode=False, batch_size=1,
                   debug=False, test_size=1):

    if visualize_mode:
        transformer = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        transformer = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]),
        ])

    loader = torch.utils.data.DataLoader(PredictListDataset(
        data_list,
        shuffle=False,
        transform=transformer
    ))

    return loader

