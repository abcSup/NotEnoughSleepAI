import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from pyntcloud import PyntCloud
from open3d import JVisualizer
from pyquaternion import Quaternion

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CAMS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]

transform = transforms.Compose([
    transforms.CenterCrop((640, 1920)),
    transforms.Resize((320, 960)),
])

def clip_points(points, xlimit=(-40, 40), ylimit=(0, 70), zlimit=(-3, 3)):
    clipped_points = points[:, (points[0, :] >= xlimit[0]) &
                               (points[0, :] <= xlimit[1]) &
                               (points[1, :] >= ylimit[0]) &
                               (points[1, :] <= ylimit[1]) &
                               (points[2, :] >= zlimit[0]) &
                               (points[2, :] <= zlimit[1]) ]

    return clipped_points

class LyftDataset(Dataset):
    def __init__(self, dataset_path):
        self.lyft_data = LyftDataset(data_path=dataset_path, json_path=dataset_path + 'data', verbose=True)
        self.sample_size = len(self.lyft_data.sample)

        self.transform = transforms.Compose([

            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.sample_size * len(CAMS)

    def __getitem__(self, item):
        sample_idx = item // len(CAMS)
        sample = self.lyft_data.sample[sample_idx]

        sd_lidar = self.lyft_data.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_lidar = self.lyft_data.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])

        pc = LidarPointCloud.from_file(self.lyft_data.data_path / sd_lidar['filename'])

        cam = CAMS[item % len(CAMS)]
        cam_token = sample['data'][cam]
        sd_cam = self.lyft_data.get('sample_data', cam_token)
        cs_cam = self.lyft_data.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])

        img = Image.open(str(self.lyft_data.data_path / cam["filename"]))
        img = transform(img)

        lidar_2_ego = transform_matrix(
            cs_lidar['translation'],
            Quaternion(cs_lidar['rotation']),
            inverse=False
        )
        ego_2_cam = transform_matrix(
            cs_cam['translation'],
            Quaternion(cs_cam['rotation']),
            inverse=True
        )
        cam_2_bev = Quaternion(axis=[1, 0, 0], angle=-np.pi / 2).transformation_matrix
        # lidar_2_cam = ego_2_cam @ lidar_2_ego
        lidar_2_bevcam = cam_2_bev @ ego_2_cam @ lidar_2_ego

        points = view_points(pc.points[:3, :], lidar_2_bevcam, normalize=False)
        points = clip_points(points)

        points = pd.DataFrame(np.swapaxes(points, 0, 1), columns=['x', 'y', 'z'])
        points = PyntCloud(points)
        voxelgrid_id = points.add_structure("voxelgrid", size_x=0.1, size_y=0.1, size_z=0.2, regular_bounding_box=False)
        voxelgrid = points.structures[voxelgrid_id]

        occ_map = voxelgrid.get_feature_vector(mode='binary')
        padded_occ = np.zeros((800, 700, 30))
        padded_occ[:occ_map.shape[0], :occ_map.shape[1], :occ_map.shape[2]] = occ_map

        return img, padded_occ