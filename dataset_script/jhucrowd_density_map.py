from comet_ml import Experiment
from joblib import Parallel, delayed
import os
import pandas as pd
import numpy as np
import scipy
import scipy.spatial
import scipy.ndimage
from PIL import Image
import h5py
from visualize_util import save_density_map
import argparse
import time
import traceback
COMET_ML_API = "S3mM1eMq6NumMxk2QJAXASkUM"
PROJECT_NAME = "crowd-counting-generate-ds"

def load_density_label(label_txt_path):
    """

    :param label_txt_path: path to txt
    :return: numpy array, p[sample, a] with a is 0 for x and 1 for y
    """
    df = pd.read_csv(label_txt_path, sep=" ", header=None)
    p = df.to_numpy()
    return p


def gaussian_filter_density(gt):
    """
    generate density map from gt
    :param gt: matrix same shape as image, where annotation label as 1
    :return:
    """
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    pts_copy = pts.copy()
    tree = scipy.spatial.KDTree(pts_copy, leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    print('total points ', len(pts))
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point

        ## try to fix OverflowError: cannot convert float infinity to integer
        sigma = np.clip(sigma, 1, 100)
        ##
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant', truncate=3)
    print('done.')
    return density

def generate_density_map(img_path, label_path, output_path):
    """

    :param img_path:
    :param label_path: txt
    :param output_path
    :return:
    """

    if os.path.exists(output_path):
        return "exist " + output_path

    gt = load_density_label(label_path)
    imgfile = Image.open(img_path).convert('RGB')
    # imgfile = image.load_img(img_path)
    img = np.asarray(imgfile)

    # empty matrix zero
    k = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density(k)
    with h5py.File(output_path, 'w') as hf:
        hf['density'] = k
    return output_path


def t_single_density_map():
    img = "/data/jhu_crowd_v2.0/val/images/0003.jpg"
    label = "/data/jhu_crowd_v2.0/val/gt/0003.txt"
    out_path = "/data/jhu_crowd_v2.0/val/unittest/0003.txt"
    out = generate_density_map(img, label, out_path)
    print(out)


def t_count(density_path, label_path):
    gt_file = h5py.File(density_path, 'r')
    target = np.asarray(gt_file['density'])
    density_count = target.sum()
    label_count = len(load_density_label(label_path))
    print("density count ", density_count)
    print("label count ", label_count)
    print("diff ", density_count - label_count)
    print("diff percentage ", (density_count - label_count)/ label_count * 100)


def t_print_density_map(density_path, density_img_out):
    gt_file = h5py.File(density_path, 'r')
    target = np.asarray(gt_file['density'])
    save_density_map(target, density_img_out)
    print("done print ", density_img_out)


def full_flow_jhucrowd(root_path, experiment=None):
    ROOT = root_path
    images_folder = os.path.join(ROOT, "images")
    gt_path_folder = os.path.join(ROOT, "ground-truth")
    density_path_folder = os.path.join(ROOT, "ground-truth-h5")
    img_list = os.listdir(path=images_folder)
    os.makedirs(density_path_folder, exist_ok=True)
    count = 0
    for img_name in img_list:
        name = img_name.split(".")[0]
        density_name = name + ".h5"
        gt_name = name + ".txt"
        if experiment is not None:
            experiment.log_metric("name", name)
        img_path = os.path.join(images_folder, img_name)
        gt_path =  os.path.join(gt_path_folder, gt_name)
        density_path = os.path.join(density_path_folder, density_name)
        out = generate_density_map(img_path, gt_path, density_path)
        print(out)
        count += 1
        if experiment is not None:
            experiment.log_metric("count", count)
    print("done")



def full_flow_jhucrowd_parallel(root_path, experiment=None):
    ROOT = root_path
    images_folder = os.path.join(ROOT, "images")
    gt_path_folder = os.path.join(ROOT, "ground-truth")
    density_path_folder = os.path.join(ROOT, "ground-truth-h5")
    img_list = os.listdir(path=images_folder)
    os.makedirs(density_path_folder, exist_ok=True)

    def jhucrowd_single_file(img_name):
        name = img_name.split(".")[0]
        density_name = name + ".h5"
        gt_name = name + ".txt"
        try:
            # if experiment is not None:
            #     experiment.log_metric("name", name)
            img_path = os.path.join(images_folder, img_name)
            gt_path = os.path.join(gt_path_folder, gt_name)
            density_path = os.path.join(density_path_folder, density_name)
            out = generate_density_map(img_path, gt_path, density_path)
            print(out)
            # if experiment is not None:
            #     experiment.log_metric("count", 1)
        except Exception as e:
            track = traceback.format_exc()
            print(track)
            # experiment.log_metric("exception_at", name)
            print("exception at ", name)


    Parallel(n_jobs=8)(delayed(jhucrowd_single_file)(img_name) for img_name in img_list)

    print("done")


# def jhucrowd_single_file(img_name):
#     # ROOT = root_path
#     # images_folder = os.path.join(ROOT, "images")
#     # gt_path_folder = os.path.join(ROOT, "ground-truth")
#     # density_path_folder = os.path.join(ROOT, "ground-truth-h5")
#     name = img_name.split(".")[0]
#     density_name = name + ".h5"
#     gt_name = name + ".txt"
#     if experiment is not None:
#         experiment.log_metric("name", name)
#     img_path = os.path.join(images_folder, img_name)
#     gt_path = os.path.join(gt_path_folder, gt_name)
#     density_path = os.path.join(density_path_folder, density_name)
#     out = generate_density_map(img_path, gt_path, density_path)
#     print(out)
#     if experiment is not None:
#         experiment.log_metric("count", 1)


def args_parser():
    """
    this is not dummy
    if you are going to make all-in-one notebook, ignore this
    :return:
    """
    parser = argparse.ArgumentParser(description='jhucrowd')
    parser.add_argument("--task_id", action="store", default="dev")
    parser.add_argument('--input', action="store",  type=str)
    arg = parser.parse_args()
    return arg


if __name__ == "__main__":
    experiment = Experiment(project_name=PROJECT_NAME, api_key=COMET_ML_API)
    start_time = time.time()
    args = args_parser()
    experiment.set_name(args.task_id)
    experiment.set_cmd_args()

    print("input ", args.input)
    full_flow_jhucrowd_parallel(args.input, experiment)
    # full_flow_jhucrowd("/data/jhu_crowd_v2.0/val")
    # t_count("/data/jhu_crowd_v2.0/val/unittest/0003.h5", "/data/jhu_crowd_v2.0/val/gt/0003.txt")
    # t_single_density_map()

    # t_print_density_map("/data/jhu_crowd_v2.0/val/ground-truth-h5/3556.h5", "/data/jhu_crowd_v2.0/val/ground-truth-h5/3556.png")
    # t_print_density_map("/data/jhu_crowd_v2.0/val/ground-truth-h5/1632.h5",
    #                     "/data/jhu_crowd_v2.0/val/ground-truth-h5/1632.png")

    # ROOT = "/data/jhu_crowd_v2.0/val"
    # images_folder = os.path.join(ROOT, "images")
    # gt_path_folder = os.path.join(ROOT, "gt")
    print("--- %s seconds ---" % (time.time() - start_time))

