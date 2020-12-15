

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os




__author__ = "Thai Thien"
__email__ = "tthien@apcs.vn"

import os
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import glob
import h5py
import time
from sklearn.externals.joblib import Parallel, delayed
import sys
from PIL import Image
import argparse
import json


def preprocess_args_parse():
    parser = argparse.ArgumentParser(description='CrowdCounting Context Aware Network')
    parser.add_argument("--root", action="store", default="dev")
    parser.add_argument("--part", action="store", default="dev")
    parser.add_argument("--output", action="store", default="dev")
    parser.add_argument("--trunc", action="store", default=4.0, type=float)
    arg = parser.parse_args()
    return arg

# where it have /images
__DATASET_ROOT = "/data/my_crowd_image/data_bike_20_q100/unsort"

# where should it write ground-truth-h5
__OUTPUT_NAME = "/data/my_crowd_image/data_bike_20_q100/unsort"

# args = preprocess_args_parse()
# __DATASET_ROOT = args.root
# __OUTPUT_NAME = args.output
__TRUNC = 4.0
# __PART = args.part


def gaussian_filter_density_fixed(gt, sigma):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        density += gaussian_filter(pt2d, sigma, mode='constant', truncate=__TRUNC)
    print('done.')
    return density


def prepare_point_from_json(json_path):
    """

    :param json_path: full json path including .json
    :return: [(x,y),(x,y)], a list of tuples
    """
    coord_list = []
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        pts = json_data['points']
        for pt in pts:
            tup = (int(pt['x']), int(pt['y']))
            coord_list.append(tup)
    return coord_list


def generate_density_map(img_path):
    print(img_path)
    json_path = img_path.replace(".jpg", ".json").replace("images", "jsons")
    print('json_path ', json_path)
    gt = prepare_point_from_json(json_path)
    imgfile = Image.open(img_path).convert('RGB')
    img = np.asarray(imgfile)
    k = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density_fixed(k, 15)
    output_path = img_path.replace(__DATASET_ROOT, __OUTPUT_NAME)\
        .replace('.jpg', '.h5')\
        .replace('images', 'ground-truth-h5')
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print("output", output_path)
    sys.stdout.flush()
    with h5py.File(output_path, 'w') as hf:
        hf['density'] = k
    return img_path


def generate_path(root):
    path_to_img = os.path.join(root, 'images')

    img_list = []

    for img_path in glob.glob(os.path.join(path_to_img, '*.jpg')):
        img_list.append(img_path)


    return img_list




if __name__ == "__main__":
    """
    generate density map from label, fixed sigma, with truncate 3
    """

    start_time = time.time()
    img_list = generate_path(__DATASET_ROOT)
    Parallel(n_jobs=10)(delayed(generate_density_map)(p) for p in img_list)



    print("--- %s seconds ---" % (time.time() - start_time))
