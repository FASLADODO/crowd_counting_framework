import os
import pandas as pd
import numpy as np
import scipy
import scipy.spatial
import scipy.ndimage
from PIL import Image
import h5py
from visualize_util import save_density_map

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
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

def generate_density_map(img_path, label_path, output_path):
    """

    :param img_path:
    :param label_path: txt
    :param output_path
    :return:
    """

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


def full_flow_jhucrowd(root_path):
    ROOT = root_path
    images_folder = os.path.join(ROOT, "images")
    gt_path_folder = os.path.join(ROOT, "gt")
    density_path_folder = os.path.join(ROOT, "ground-truth-h5")
    img_list = os.listdir(path=images_folder)
    os.makedirs(density_path_folder, exist_ok=True)
    for img_name in img_list[:3]:
        name = img_name.split(".")[0]
        density_name = name + ".h5"
        gt_name = name + ".txt"
        img_path = os.path.join(images_folder, img_name)
        gt_path =  os.path.join(gt_path_folder, gt_name)
        density_path = os.path.join(density_path_folder, density_name)
        out = generate_density_map(img_path, gt_path, density_path)
        print(out)
    print("done")


if __name__ == "__main__":
    # full_flow_jhucrowd("/data/jhu_crowd_v2.0/val")
    # t_count("/data/jhu_crowd_v2.0/val/unittest/0003.h5", "/data/jhu_crowd_v2.0/val/gt/0003.txt")
    # t_single_density_map()

    t_print_density_map("/data/jhu_crowd_v2.0/val/ground-truth-h5/3556.h5", "/data/jhu_crowd_v2.0/val/ground-truth-h5/3556.png")
    t_print_density_map("/data/jhu_crowd_v2.0/val/ground-truth-h5/1632.h5",
                        "/data/jhu_crowd_v2.0/val/ground-truth-h5/1632.png")

    # ROOT = "/data/jhu_crowd_v2.0/val"
    # images_folder = os.path.join(ROOT, "images")
    # gt_path_folder = os.path.join(ROOT, "gt")

