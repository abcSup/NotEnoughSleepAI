from os.path import join
import random
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset

class KITTI_DepthCompletion(Dataset):
    def __init__(self, kitti_dir, data_depth_annotated_dir, data_depth_velodyne_dir, data_RGB_dir, size, test=False):

        self._size = size
        self._test = test
        self._kitti_dir = kitti_dir #KITTI parent directory
        self._data_depth_annotated_dir = data_depth_annotated_dir #Ground truth depth map
        self._data_depth_velodyne_dir = data_depth_velodyne_dir #Projected LIDAR
        self._data_RGB_dir = data_RGB_dir #RGB
        self._image_transform = transforms.Compose([
            transforms.Resize(self._size),
            transforms.ToTensor(),
            ])
        with open(self._data_depth_annotated_dir, 'r') as text:
            self._depth_map = text.readlines()

        with open(self._data_depth_velodyne_dir, 'r') as text:
            self._lidar = text.readlines()

        with open(self._data_RGB_dir, 'r') as text:
            self._rgb = text.readlines()

    def __len__(self):

        return len(self.images_dir)

    def __getitem__(self, idx):

        #disp(u,v)  = ((float)I(u,v))/256.0;
        #valid(u,v) = I(u,v)>0;
        #For lidar and depth_map, depth by disparity can be calculated via disp(u, v)
        #Some pixels have the value 0, which means there is no G.T. available, thus should not be calculated for loss
        #Valid pixels can be acquired via valid(u, v)

        depth_map = Image.open(join(self._kitti_dir, self._depth_map[idx]))
        lidar = Image.open(join(self._kitti_dir, self._lidar[idx]))
        rgb = Image.open(join(self._kitti_dir, self._rgb[idx]))

        #if not self.test:
            #image_l, image_r = image_augmentation(image_l, image_r)

        depth_map = self._image_transform(depth_map)
        lidar = self._image_transform(lidar)
        rgb = self._image_transform(rgb)

        return (depth_map, lidar, rgb)

def image_augmentation(left_img, right_img):
    random_gamma = random.uniform(0.8, 1.2)
    random_brightness = random.uniform(0.5, 2.0)
    random_flip = random.uniform(0.0, 1.0)

    TF.adjust_gamma(left_img, random_gamma)
    TF.adjust_gamma(right_img, random_gamma)

    TF.adjust_brightness(left_img, random_brightness)
    TF.adjust_brightness(right_img, random_brightness)

    if random_flip > 0.5:
        temp_img = TF.hflip(left_img)
        right_img = TF.hflip(right_img)
        left_img = right_img
        right_img = temp_img

    return left_img, right_img