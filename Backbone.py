import torch.nn as nn
from .datasets.data_channels import LidarChannels, ImageChannels

class Backbone(nn.Module):
    
    def __init__(self, bev_height, img_chn, kNN=1):
        super(Backbone, self).__init__()
        #NOTE: FOR BEV_BLOCK, ALL CONV2D MUST BE DONE WITH GROUPS=IN_CHN, AND IN_CHN=OUT_CHN
        #       THIS IS TO MAINTAIN BEV POINTS CORRESPONDENCE, HEIGHT-WISE
        #BLOCK1
        self.bev_block1 = bev_backbone(in_chn=bev_height, out_chn=bev_height)
        self.fusion_block1 = densefusion(k=kNN)
        self.image_block1 = image_backbone(in_chn=img_chn, out_chn=64)

        #BLOCK2
        self.bev_block2 = bev_backbone(in_chn=bev_height, out_chn=bev_height)
        self.fusion_block2 = densefusion(k=kNN)
        self.image_block2 = image_backbone(in_chn=64, out_chn=128)

        #BLOCK3
        self.bev_block3 = bev_backbone(in_chn=bev_height, out_chn=bev_height)
        self.fusion_block3 = densefusion(k=kNN)
        self.image_block3 = image_backbone(in_chn=128, out_chn=256)

        #BLOCK4
        self.bev_block4 = bev_backbone(in_chn=bev_height, out_chn=bev_height)
        self.fusion_block4 = densefusion(k=kNN)
        self.image_block4 = image_backbone(in_chn=256, out_chn=512)

        #FINAL OUTPUT BLOCKS
        self.bev_block5 = upconv_block(in_chn=bev_height, out_chn=bev_height, factor=4)
        self.image_block5 = upconv_block(in_chn=512, out_chn=512, factor=4)

    def forward(self, sample):

        #Generate neccesary data
        sample.map_pointcloud_to_images()
        sample.corresponding_lidar.voxelize()
        sample.corresponding_lidar.generateKNN()
        bev_input = sample.corresponding_lidar.getOccupancyMatrix()

        #TODO Make sure all inputs are transformed into tensor before passing into NN
        #BLOCK1
        im1 = {}
        for chn in ImageChannels:
            im_input = sample.corresponding_images.getImage(chn)
            sparse_input = sample.getMappedLidar(chn)
            im1[chn] = self.image_block1(im_input, sparse_input)

        fused1 = self.fusion_block1(im1, bev_input, sample)
        bev1 = self.bev_block1(fused1)

        #BLOCK2
        im2 = {}
        for chn in ImageChannels:
            im2[chn] = self.image_block2(im1[chn])

        fused2 = self.fusion_block2(im2, bev1, sample)
        bev2 = self.bev_block2(fused2)

        #BLOCK3
        im3 = {}
        for chn in ImageChannels:
            im3[chn] = self.image_block3(im2[chn])

        fused3 = self.fusion_block3(im3, bev2, sample)
        bev3 = self.bev_block3(fused3)

        #BLOCK4
        im4 = {}
        for chn in ImageChannels:
            im4[chn] = self.image_block4(im3[chn])
        fused4 = self.fusion_block4(im4, bev3, sample)
        bev4 = self.bev_block4(fused4)


        im5 = {}
        for chn in ImageChannels:
            im5[chn] = self.image_block5(im4[chn])
        bev5 = self.bev_block5(bev4)

        return im5, bev5

