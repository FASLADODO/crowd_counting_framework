import glob
import PIL.Image as Image
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import os
import numpy as np

from PIL import Image


def save_density_map(density_map, name):
    plt.figure(dpi=600)
    plt.axis('off')
    plt.margins(0, 0)
    plt.imshow(density_map, cmap=CM.jet)
    plt.savefig(name, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_density_map_with_colorrange(density_map, name, vmin, vmax):
    plt.figure(dpi=600)
    plt.axis('off')
    plt.margins(0, 0)
    plt.imshow(density_map, cmap=CM.jet)
    plt.clim(vmin, vmax)
    plt.savefig(name, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_img(imgnp, name):
    # plt.imshow(imgnp[0].permute(1, 2, 0).numpy())
    plt.imsave(name, imgnp[0].permute(1, 2, 0).numpy())
    # plt.show()
    # im = Image.fromarray(imgnp[0].permute(1, 2, 0).numpy())
    # im.save(name)
