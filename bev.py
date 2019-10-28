import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from pyntcloud import PyntCloud
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

s_time = time.time()
#dataset = LyftDataset(data_path='.', json_path='data/', verbose=True)
lidar = LidarPointCloud.from_file(Path("lidar/host-a004_lidar1_1232815252301696606.bin"))
points = lidar.points
print(points.shape)

axes_xlimit = 70
axes_ylimit = 40

x = points[:3, :]
x = np.swapaxes(x, 0, 1)
#x = x[np.where(([0, -40] <= x[:, :2]) & (x[:, :2] <= [70, 40]))]
x = x[(x[:, 0] >= 0) & (x[:, 0] <= 70) & (x[:, 1] >= -40) & (x[:, 1] <= 40)]
print(x.shape)


df = pd.DataFrame(x, columns=['x', 'y', 'z'])
print(df.describe())
cloud = PyntCloud(df)

voxelgrid_id = cloud.add_structure("voxelgrid", n_x=512, n_y=448, n_z=32)
voxelgrid = cloud.structures[voxelgrid_id]
density_feature_vector = voxelgrid.get_feature_vector(mode="density")
print(density_feature_vector.shape)

new_cloud = cloud.get_sample("voxelgrid_nearest", n=8, voxelgrid_id=voxelgrid_id, as_PyntCloud=True)
#new_cloud.plot()
print(new_cloud)

e_time = time.time()
print(e_time - s_time)
"""
x_points = x[:, 0]
y_points = x[:, 1]
_, ax = plt.subplots(1, 1, figsize=(7, 8))
ax.scatter(x_points, y_points, s=0.2)
"""

#plt.show()
