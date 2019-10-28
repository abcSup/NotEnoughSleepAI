import torch.nn as nn

class Backbone(nn.Module):
    
    def __init__(self, bev_height, img_chn, kNN=1):
        super(Backbone, self).__init__()
        #BLOCK1
        self.bev_block1 = bev_backbone(in_chn=bev_height, out_chn=64)
        self.fusion_block1 = densefusion(k=kNN)
        self.image_block1 = image_backbone(in_chn=img_chn, out_chn=64)

        #BLOCK2
        self.bev_block2 = bev_backbone(in_chn=64, out_chn=128)
        self.fusion_block2 = densefusion(k=kNN)
        self.image_block2 = image_backbone(in_chn=64, out_chn=128)

        #BLOCK3
        self.bev_block3 = bev_backbone(in_chn=128, out_chn=256)
        self.fusion_block3 = densefusion(k=kNN)
        self.image_block3 = image_backbone(in_chn=128, out_chn=256)

        #BLOCK4
        self.bev_block4 = bev_backbone(in_chn=256, out_chn=512)
        self.fusion_block4 = densefusion(k=kNN)
        self.image_block4 = image_backbone(in_chn=256, out_chn=512)

        #FINAL OUTPUT BLOCKS
        self.bev_block5 = upconv_block(in_chn=512, out_chn=512, factor=4)
        self.image_block5 = upconv_block(in_chn=512, out_chn=512, factor=4)

    def forward(self, im_input, bev_input):
        im1 = self.image_block1(im_input)
        fused1 = self.fusion_block1(im1, bev_input)
        bev1 = self.bev_block1(fused1)

        im2 = self.image_block2(im1)
        fused2 = self.fusion_block2(im2, bev1)
        bev2 = self.bev_block2(fused2)

        im3 = self.image_block3(im2)
        fused3 = self.fusion_block3(im3, bev2)
        bev3 = self.bev_block3(fused3)

        im4 = self.image_block4(im3)
        fused4 = self.fusion_block4(im4, bev3)
        bev4 = self.bev_block4(fused4)

        im5 = self.image_block5(im4)
        bev5 = self.bev_block5

        return im5, bev5

