import torch.nn as nn
from datasets.data_channels import LidarChannels, ImageChannels

class Backbone(nn.Module):
    
    def __init__(self, bev_height, img_chn, kNN=1):
        super(Backbone, self).__init__()
        #NOTE: FOR BEV_BLOCK, ALL CONV2D MUST BE DONE WITH GROUPS=IN_CHN, AND IN_CHN=OUT_CHN
        #       THIS IS TO MAINTAIN BEV POINTS CORRESPONDENCE, HEIGHT-WISE
        #BLOCK1
        self.bev_block1 = BEVBlock(in_chn=bev_height, out_chn=bev_height)
        self.fusion_block1 = DenseFusionBlock(k=kNN)
        self.image_block1 = ImageBlock(in_chn=img_chn, out_chn=64)

        #BLOCK2
        self.bev_block2 = BEVBlock(in_chn=bev_height, out_chn=bev_height).registerPreviousSampleable(self.bev_block1)
        self.fusion_block2 = DenseFusionBlock(k=kNN)
        self.image_block2 = ImageBlock(in_chn=64, out_chn=128).registerPreviousSampleable(self.image_block1)

        #BLOCK3
        self.bev_block3 = BEVBlock(in_chn=bev_height, out_chn=bev_height).registerPreviousSampleable(self.bev_block2)
        self.fusion_block3 = DenseFusionBlock(k=kNN)
        self.image_block3 = ImageBlock(in_chn=128, out_chn=256).registerPreviousSampleable(self.image_block2)

        #BLOCK4
        self.bev_block4 = BEVBlock(in_chn=bev_height, out_chn=bev_height).registerPreviousSampleable(self.bev_block3)
        self.fusion_block4 = DenseFusionBlock(k=kNN)
        self.image_block4 = ImageBlock(in_chn=256, out_chn=512).registerPreviousSampleable(self.image_block3)

        #FINAL OUTPUT BLOCKS
        self.bev_block5 = UpConvBlock(in_chn=bev_height, out_chn=bev_height, factor=4)
        self.image_block5 = UpConvBlock(in_chn=512, out_chn=512, factor=4)

    def forward(self, sample):

        #Generate neccesary data
        sample.map_pointcloud_to_images()
        sample.corresponding_lidar.voxelize()
        sample.corresponding_lidar.generateKNN()
        bev_input = sample.corresponding_lidar.getOccupancyMatrix()

        #TODO Make sure all inputs are transformed into tensor before passing into NN
        #BLOCK1
        im_input = {}
        im1 = {}
        for chn in ImageChannels:
            sparse_input = sample.getMappedLidar(chn)
            im_input[chn] = sample.corresponding_images.getImage(chn)
            im1[chn] = self.image_block1(im_input[chn], sparse_input)

        fused1 = self.fusion_block1(im_input, bev_input, sample)
        bev1 = self.bev_block1(fused1)

        #BLOCK2
        im2 = {}
        for chn in ImageChannels:
            im2[chn] = self.image_block2(im1[chn])

        fused2 = self.fusion_block2(im1, bev1, sample)
        bev2 = self.bev_block2(fused2)

        #BLOCK3
        im3 = {}
        for chn in ImageChannels:
            im3[chn] = self.image_block3(im2[chn])

        fused3 = self.fusion_block3(im2, bev2, sample)
        bev3 = self.bev_block3(fused3)

        #BLOCK4
        im4 = {}
        for chn in ImageChannels:
            im4[chn] = self.image_block4(im3[chn])
        fused4 = self.fusion_block4(im3, bev3, sample)
        bev4 = self.bev_block4(fused4)


        im5 = {}
        for chn in ImageChannels:
            im5[chn] = self.image_block5(im4[chn])
        bev5 = self.bev_block5(bev4)

        return im5, bev5

class SampleableBlock(nn.Module):
    def __init__(self):
        self._previousSampleable = None

    def calculateReceptiveField(self, size, jump, receptive_field, start):
        if (self._previousSampleable):
            size, jump, receptive_field, start = self._previousSampleable.calculateReceptiveField(size, jump, receptive, start)

        #Calculations used from https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
        new_size = self.getNewSize(size)
        new_jump = self.getNewJump(jump)
        new_receptive_field = self.getNewReceptiveField(receptive_field, jump)
        new_start = self.getNewStart(start, size, jump)

        return new_size, new_jump, new_receptive_field, new_start 

    def getNewSize(self, size):
        layers = [module for module in self.modules() if type(module) != nn.Sequential]
        for layer in layers:
            if all(hasattr(layer, attr) for attr in ["kernel_size", "padding", "stride"]):
                k = layer.kernel_size
                p = layer.padding
                s = layer.stride

                #Calculation
                size = math.floor((size - k + 2 * p) / s) + 1

        return size

    def getNewJump(self, jump):
        layers = [module for module in self.modules() if type(module) != nn.Sequential]
        for layer in layers:
            if all(hasattr(layer, attr) for attr in ["stride"]):
                s = layer.stride

                #Calculation
                jump = jump * s

        return jump

    def getNewReceptiveField(self, receptive_field, jump):
        layers = [module for module in self.modules() if type(module) != nn.Sequential]
        for layer in layers:
            if all(hasattr(layer, attr) for attr in ["kernel_size", "stride"]):
                k = layer.kernel_size
                s = layer.stride

                #Calculation
                receptive_field = receptive_field + (k - 1) * jump
                jump = jump * s

        return receptive_field

    def getNewStart(self, start, size, jump):
        layers = [module for module in self.modules() if type(module) != nn.Sequential]
        for layer in layers:
            if all(hasattr(layer, attr) for attr in ["kernel_size", "stride", "padding"]):
                k = layer.kernel_size
                p = layer.padding
                s = layer.stride

                #Calculation
                size_out = math.floor((size - k + 2 * p) / s) + 1
                actualP = (size_out - 1) * s - size + k 
                pL = math.floor(actualP / 2)
                start = start + ((k - 1) / 2 - pL) * jump
                jump = jump * s

                #Update for next iteration, don't get confused
                size = size_out

        return start

    def registerPreviousSampleable(self, prev_sampleable):
        assert isinstance(prev_sampleable, SampleableBlock), "Only a SampleableBlock can be registered as the predecessor of another SampleableBlock"
        self._previousSampleable = prev_sampleable


#TODO Modularize every block
class BEVBlock(SampleableBlock):
    def __init__(self):
        pass

class ImageBlock(SampleableBlock):
    def __init__(self):
        pass

class UpConvBlock(SampleableBlock):
    def __init__(self):
        pass
