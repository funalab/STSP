import os
import numpy as np
import random
import skimage.io as io
from skimage import transform
from skimage.color import rgb2gray
from glob import glob

import torch
from torch.utils.data import Dataset

def min_max_normalize_one_image(image):
    """
    normalize the itensity of an nd image based on the MIN-MAX normalization [0, 1]
    inputs:
        volume: the input nd image
    outputs:
        out: the normalized nd image
    """

    max_int = image.max()
    min_int = image.min()
    out = (image - min_int) / (max_int - min_int)

    return out

def zscore_normalize_one_image(image):
    """
    normalize the itensity of an nd image based on the z-score normalization
    inputs:
        volume: the input nd image
    outputs:
        out: the normalized nd image
    """

    mean_int = image.mean()
    std_int = image.std()
    out = (image - mean_int) / std_int

    return out

def crop_2d(
        image,
        crop_size=(1024, 1024),
        augmentation=True
    ):
    """ 2d RGB image patche is cropped from array.
    Args:
        image (np.ndarray)                  : Input 2d image array
        crop_size ((int, int))              : Crop image patch from array randomly
        nb_crop (int)                       : Number of cropping patches at once
    """
    _, y_len, x_len = image.shape
    assert x_len >= crop_size[1]
    assert y_len >= crop_size[0]

    cropped_image = []

    if augmentation:
        # get cropping position (image)
        top = random.randint(0, x_len-crop_size[1]-1) if x_len > crop_size[1] else 0
        left = random.randint(0, y_len-crop_size[0]-1) if y_len > crop_size[0] else 0
        bottom = top + crop_size[1]
        right = left + crop_size[0]
        cropped_image = np.array(image[:, left:right, top:bottom])

        # augmentation image rotation & flip
        rot_flag = random.randint(0, 4)
        flip_flag = random.randint(0, 2)
        for c in range(len(cropped_image)):
            cropped_image[c] = np.rot90(cropped_image[c], k=rot_flag)
            if flip_flag:
                cropped_image[c] = np.flip(cropped_image[c], axis=0)

    else:
        top = int((x_len - crop_size[1])/2)
        left = int((y_len - crop_size[0])/2)
        bottom = top + crop_size[1]
        right = left + crop_size[0]
        cropped_image = np.array(image[:, left:right, top:bottom])

    return cropped_image


class STDataset(Dataset):
    def __init__(self, root=None, split_list=None, label_list=None, basename='images', crop_size=[1024, 1024], train=True):
        self.root = root
        self.basename = basename
        self.train = train
        self.crop_size = crop_size
        with open(split_list, 'r') as f:
            self.file_list = [line.rstrip() for line in f]
        with open(label_list, 'r') as f:
            self.label_list = [line.rstrip() for line in f]

    def __len__(self):
        return len(self.file_list)

    def get_image(self, i):
        image = io.imread(os.path.join(self.root, self.basename, self.file_list[i])).transpose(2, 0, 1)
        image = crop_2d(image, crop_size=self.crop_size, augmentation=self.train)
        image = min_max_normalize_one_image(image)
        return image

    def get_label(self, i):
        label = self.label_list.index(self.file_list[i][self.file_list[i].rfind('_')+1:self.file_list[i].rfind('-')])
        return np.array([label])

    def __getitem__(self, i):
        image, label = self.get_image(i), self.get_label(i)
        return torch.tensor(image).float(), torch.tensor(label).long()
