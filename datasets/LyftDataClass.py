import copy
import torch
import numpy as np
from lyft_dataset_sdk.lyftdataset import LyftDataset
from .data_channels import LidarChannels, ImageChannels
from PIL import Image
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
from pyquaternion import Quaternion

class LyftDataClass(Dataset):
    def __init__(self, data_path, json_path, verbose=True, map_resolution=0.1):
        self._lyftdataset = LyftDataset(data_path, json_path, verbose=verbose, map_resolution=map_resolution)

    def __len__(self):
        return len(self._lyftdataset.scene)

    def __getitem__(self, idx):
        sample_token = self._lyftdataset.sample[idx]
        sample_record = self.getFromLyft("sample", sample_token)

        return Sample(self, sample_record)

    def getFromLyft(self, table_name, token):
        return self._lyftdataset.get(table_name, token)

    def getFromLyftDatapath(self):
        return self._lyftdataset.data_path


class Sample:
    def __init__(self, lyftdataclass, sample_record, dense=False):
        self._lyftdataclass = lyftdataclass

        #if dense:
            #TODO if dense=True, merge LIDAR_TOP, LIDAR_FRONT_LEFT, LIDAR_FRONT_RIGHT to form a dense point cloud

        #Initialize CorrespondingLidarPointCloud
        pointsensor_token = sample_record["data"][LidarChannels.LIDAR_TOP]
        pointsensor = self._lyftdataclass.getFromLyft("sample_data", self._pointsensor_token)
        pcl_path = self._lyftdataclass.getFromLyftDatapath() / pointsensor["filename"]
        self.__init_LidarPointCloud__(pcl_path, pointsensor)

        #Initialize CorrespondingImages
        images = {}
        for chn in ImageChannels:
            camera_token = [sample_record["data"][chn.value]]
            cam = self._lyftdataclass.getFromLyft("sample_data", camera_token)
            im = Image.open(str(self._lyftdataclass.getFromLyftDatapath() / cam["filename"]))
            images[chn] = (im, cam)
        self.__init_Images__(images)

    def __init_LidarPointCloud__(self, pcl_path, pointsensor):
        self.corresponding_lidar = CorrespondingLidarPointCloud(pcl_path, pointsensor)

    def __init_Images__(self, images):
        self.corresponding_images = CorrespondingImages(images)

    def getMappedLidar(self, chn):
        assert ImageChannels.hasValue(chn), "Invalid channel, must be a value in ImageChannels"
        assert self._mapped_lidar, "Call map_pointcloud_to_images() first"

    #Ported from nuscenes-devkit
    def map_pointcloud_to_images(self):

        self._mapped_lidar = {}
        for chn in ImageChannels:
            pc = copy.deepcopy(self.corresponding_lidar)
            im = self.corresponding_images.getImage(chn)
            cam = self.corresponding_images.getCam(chn)
            pointsensor = self.corresponding_images.getPointsensor()

            # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
            # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
            cs_record = self._lyftdataclass.getFromLyft("calibrated_sensor", pointsensor["calibrated_sensor_token"])
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform to the global frame.
            poserecord = self._lyftdataclass.getFromLyft("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform into the ego vehicle frame for the timestamp of the image.
            poserecord = self._lyftdataclass.getFromLyft("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform into the camera.
            cs_record = self._lyftdataclass.getFromLyft("calibrated_sensor", cam["calibrated_sensor_token"])
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Retrieve the color from the depth.
            coloring = depths

            # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
            points = view_points(pc.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

            # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > 0)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
            points = points[:, mask]
            coloring = coloring[mask]

            #TODO Convert points to sparse lidar map, same size as im
            sparse = np.zeros((im.size[0], im.size[1]))
            _p = np.swapaxes(points[:2, :], 0, 1).astype(np.int)
            sparse[(tuple(_p[:, 0]), tuple(_p[:, 1]))] = depths[mask]

            self._mapped_lidar[chn] = (points, mask, coloring, sparse)

class CorrespondingLidarPointCloud():
    def __init__(self, pcl_path, pointsensor):
        self._points = np.fromfile(str(pcl_path), dtype=np.float32).reshape((-1, 5))[:, :3]
        self._pointsensor = pointsensor
        self._pc = PyntCloud(self._points.T) #[3, N] -> [N, 3]

    def getPointCloud(self):
        return self._points

    def getPointsensor(self):
        return self._pointsensor

    def getOccupancyMatrix(self):
        assert self._occupancy, "Call voxelize() first"
        return self._occupancy

    def getNeighbors(self):
        assert self._neighbors, "Call generateKNN() first"
        return self._neighbors

    def getKNN(self, k):
        return self._pc.get_neighbors(k=k)

    def nbr_points(self) -> int:
        return self._points.shape[1]

    def voxelize(self, size_x, size_y, size_z):
        voxelgrid_id = self._pc.add_structure("voxelgrid", size_x=size_x, size_y=size_y, size_z=size_z, regular_bounding_box=False)
        voxelgrid = self._pc.structures[voxelgrid_id]
        self._occupancy = voxelgrid.get_feature_vector(mode='binary')

    def generateKNN(self):
        self._neighbors = self._pc.get_neighbors() #(N, k)

    def translate(self, x):
        for i in range(3):
            self._points[i, :] = self._points[i, :] + x[i]

    def rotate(self, rot_matrix):
        self._points = np.dot(rot_matrix, self._points)

    def transform(self, transf_matrix):
        self._points = transf_matrix.dot(np.vstack((self._points, np.ones(self.nbr_points()))))


class CorrespondingImages():
    def __init__(self, images):
        self._images = images

    def getImage(self, chn):
        assert ImageChannels.hasValue(chn), "Invalid channel, must be a value in ImageChannels"
        return self._images[chn][0]

    def getCam(self, chn):
        assert ImageChannels.hasValue(chn), "Invalid channel, must be a value in ImageChannels"
        return self._images[chn][1]


#TODO Define class for ground truth
class CorrespondingGroundTruth():
    pass
