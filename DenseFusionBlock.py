import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import numpy as np

class DenseFusionBlock(nn.Module):
    
    #Selection criterions for higher-level (downsampled) features to find determine correspondence among multiple candidates,
    #   as each pixel in a higher-level feature map represents a group of pixels in the original input 
    #Selection criterions: interpolate (bilinear interpolation), maxpool, avgpool

    def __init__(self, rotation_matrix, translation_vector, layers=4, k=1, selection_criterion='interpolate'):

        self._projection_matrix = projection_matrix
        self._layers = layers
        self._k = k
        self._selection_criterion = selection_criterion

        linear_layers = []
        linear_layers += [nn.Linear(k, k, bias=True)]
        linear_layers += [nn.Linear(k, 1, bias=True)]
            
        self._linear_block_i = nn.Sequential(*linear_layers)

    '''
    Method:
        forward: Takes in image and BEV feature maps, perform continuous fusion as in Deep Continuous Fusion for Multi-Sensor 3D
                Object Detection (Liang et. al.).
                For each pixel in BEV feature map, the corresponding LIDAR points is acquired.
                    The point is then projected on to the image plane, to acquire the corresponding image (geometric) feature.
                    The image feature is then put through an MLP (Multi-layer Perceptron) network to compute a final image output feature
                The final image output feature has a one-to-one correspondence with the original, unprocessed image feature input.
                Then, the method outputs BEV feature + image output feature (concatenation) as the fused feature.

    Parameters:
        image: dict of torch.Tensor
            (key, value) pairs of (ImageChannel, (N * C * H * W) torch.Tensor of image (geometric) feature)
        bev: torch.Tensor
            (N * C * H * W) BEV feature
        sample: Sample object
            An instance of the Sample class containing image and bev
    
    Return(s):
        fused_feature: (N * C * H * W) image + BEV feature (concatenated)
    '''
    def forward(self, image, depth, bev):
        #TODO Optimize
        n_points = []
        #UNPROJECT IMAGE TO METRIC 3D SPACE
        for n in range(image.shape[0]):
            points = []
            for x, y in zip(*np.where(depth[n, 0, :, :] > 0)):
                #cx_d, cy_d, fx_y, fx_d are the intrinsics
                P3D_x = (x - cx_d) * depth[n, 0, x, y] / fx_d
                P3D_y = (y - cy_d) * depth[n, 0, x, y] / fy_d
                P3D_z = depth[n, 0, x, y]

                points += [(np.array(P3D_x, P3D_y, P3D_z), (np.array(image[n, :, x, y])))] #Get all points and their corresponding RGB values

            n_points += points


        #TRANSFORM TO LIDAR SENSOR COORDINATES

        #SUM PIXEL-WISE WITH BEV FEATURE

        return fused_feature



