import torch.nn as nn
from datasets.data_channels import LidarChannels, ImageChannels

class Backbone(nn.Module):
    
    def __init__(self, bev_height, img_chn, num_classes):
        super(Backbone, self).__init__()
        #BLOCK1
        self.bev_block1 = BEVBlock(bev_height, 64, 1, 1)
        self.fusion_block1 = DenseFusionBlock()
        self.image_block1 = ImageBlock(img_chn, 64, 1, 1)
        self.depth_block1 = DepthBlock(img_chn, 64, 1, 1)

        #BLOCK2
        self.bev_block2 = BEVBlock(64, 128, 1, 1)
        self.fusion_block2 = DenseFusionBlock()
        self.image_block2 = ImageBlock(64, 128, 1, 1)
        self.depth_block2 = DepthBlock(64, 128, 1, 1)

        #BLOCK3
        self.bev_block3 = BEVBlock(128, 256, 1, 1)
        self.fusion_block3 = DenseFusionBlock()
        self.image_block3 = ImageBlock(128, 256, 1, 1)
        self.depth_block3 = DepthBlock(128, 256, 1, 1)

        #BLOCK4
        self.bev_block4 = BEVBlock(256, 512, 1, 1)
        self.fusion_block4 = DenseFusionBlock()
        self.image_block4 = ImageBlock(256, 512, 1, 1)
        self.depth_block4 = DepthBlock(256, 512, 1, 1)

        #FINAL OUTPUT BLOCKS
        self.bev_block5 = UpConvBlock(512, 512, factor=4)
        self.image_block5 = UpConvBlock(512, 512, factor=4)

        #HEADER
        self.header = Header(num_classes)

    def forward(self, sample, img_chn):
        assert ImageChannels.hasValue(chn), "Invalid channel, must be a value in ImageChannels"

        #Generate neccesary data
        sample.corresponding_lidar.voxelize()
        im_input = sample.corresponding_images.getImage(img_chn)
        sparse_input = sample.getMappedLidar(img_chn, 'SPARSE')
        bev_input = sample.corresponding_lidar.getOccupancyMatrix() 

        #Get X, Y dimensions of BEV (for pixel correspondence) MUST BE SQUARE
        bev_size = bev_input.shape[2:]
        im_size = im_input.shape[2:]
        sparse_size = sparse_input.shape[2:]
        assert bev_size[0] == bev_size[1], "BEV slices (of size {}) must be squares (calculateReceptiveField() makes this assumption)".format(bev_size)
        assert im_size == sparse_size, "Image (of size {}) != sparse depth map (of size {})".format(im_input.shape, sparse_input.shape)

        im_sparse_concat_input = torch.cat((im_input, bev_input), axis=1)

        #TODO Initial and final conv layers
        #BLOCK1     
        im1 = self.image_block1(im_sparse_concat_input)
        sparse1 = self.depth_block1(sparse_input)
        fused1 = self.fusion_block1(im_input, sparse_input, bev_input)
        bev1 = self.bev_block1(fused1)

        #BLOCK2
        im2 = self.image_block2(im1)
        sparse2 = self.depth_block2(sparse1)
        fused2 = self.fusion_block2(im1, sparse1, bev1)
        bev2 = self.bev_block2(fused2)

        #BLOCK3
        im3 = self.image_block3(im2)
        sparse3 = self.depth_block3(sparse2)
        fused3 = self.fusion_block3(im2, sparse2, bev2)
        bev3 = self.bev_block3(fused3)

        #BLOCK4
        im4 = self.image_block4(im3)
        sparse4 = self.depth_block4(sparse3)
        fused4 = self.fusion_block4(im3, sparse3, bev3)
        bev4 = self.bev_block4(fused4)

        im5 = self.image_block5(im4)
        bev5 = self.bev_block5(bev4)

        header_out = self.header(bev5)

        #TODO 3D Bounding-box refinement 
        return im5, header_out

class SampleableBlock(nn.Module):
    def __init__(self):
        self._previousSampleable = None

    def calculateReceptiveField(self, size):
        if (self._previousSampleable):
            size, jump, receptive_field, start = self._previousSampleable.calculateReceptiveField(size)

        else: #This is the first layer, use initial values
            jump = 1
            receptive_field = 1
            start = 0.5

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


def conv3x3(in_chn, out_chn, stride=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_chn, out_chn, stride=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=stride, bias=False)

class BEVBlock(SampleableBlock):
    def __init__(self, in_chn, dim_size, num_blocks, stride=1):
        super(SampleableBlock, self).__init__()

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlock(in_chn, dim_size, stride)]
            in_chn = dim_size * 1
        
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class ImageBlock(SampleableBlock):
    def __init__(self, in_chn, dim_size, num_blocks, stride=1):
        super(SampleableBlock, self).__init__()

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlock(in_chn, dim_size, stride)]
            in_chn = dim_size * 1
        
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class DepthBlock(SampleableBlock):
    def __init__(self, in_chn, dim_size, num_blocks, stride=1):
        super(SampleableBlock, self).__init__()

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlock(in_chn, dim_size, stride)]
            in_chn = dim_size * 1
        
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class UpConvBlock(SampleableBlock):
    def __init__(self):
        pass

class Header(nn.Module):
    def __init__(self, num_classes):
        super(DetectionHeader, self).__init__()

        self.anchor_orients = [0, np.pi/2]
        self.score_out = (num_classes + 1) * len(self.anchor_orients)
        # (t, dx, dy, dz, l, w, h) * 2 anchors
        self.bbox_out = 8 * len(self.anchor_orients)

        self.conv1 = nn.Conv2d(256, self.score_out + self.bbox_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        clsscore_bbox = self.conv1(x)
        cls_score, bbox = torch.split(clsscore_bbox, [self.score_out, self.bbox_out], dim=1)

        return cls_score, bbox

class BasicBlock(nn.Module):
    def __init__(self, in_chn, dim_size, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_chn, dim_size, stride)
        self.bn1 = nn.BatchNorm2d(dim_size)
        self.conv2 = conv3x3(dim_size, dim_size * 1)
        self.bn2 = nn.BatchNorm2d(dim_size)
        self.activation = nn.ReLU(inplace=True)

        self.downsample = None
        if stride == 2:
            layers = []
            layers += [conv1x1(in_chn, dim_size, stride)]
            layers += [nn.BatchNorm2d(dim_size)]
            self.downsample = nn.Sequential(*layers)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.activation(out)

        return out
