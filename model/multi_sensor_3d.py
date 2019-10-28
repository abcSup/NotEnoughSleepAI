import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def conv3x3(in_chn, out_chn, stride=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_chn, out_chn, stride=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=stride, bias=False)


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


class ResNet18FPN(nn.Module):
    def __init__(self):
        super(ResNet18FPN, self).__init__()

        self.in_chn = 64
        self.conv1 = nn.Conv2d(3, self.in_chn, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_chn)
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.lateral3 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.lateral1 = nn.Conv2d(64, 256, kernel_size=1, stride=1)

    def _make_layer(self, dim_size, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlock(self.in_chn, dim_size, stride)]
            self.in_chn = dim_size * 1
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        c1 = self.activation(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.lateral3(c4))
        p3 = self._upsample_add(p4, self.lateral2(c3))
        p2 = self._upsample_add(p3, self.lateral1(c2))

        # No final smoothing layer like FPN

        return p2

class BEVBackbone(nn.Module):
    def __init__(self):
        super(BEVBackbone, self).__init__()

        self.in_chn = 32
        self.conv1 = nn.Conv2d(3, self.in_chn, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_chn)
        self.conv2 = nn.Conv2d(self.in_chn, self.in_chn, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.in_chn)
        self.activation = nn.ReLU(inplace=True)

        # We use BasicBlocks here, but Deep Continuous Fusion uses ResBlock
        self.layer1 = self._make_layer(64, 4, stride=2)
        self.layer2 = self._make_layer(128, 8, stride=2)
        self.layer3 = self._make_layer(192, 12, stride=2)
        self.layer4 = self._make_layer(256, 12, stride=2)

        self.toplayer = nn.Conv2d(256, 256, kernel_size=1, stride=1)

        self.lateral3 = nn.Conv2d(192, 256, kernel_size=1, stride=1)
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1, stride=1)

    def _make_layer(self, dim_size, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlock(self.in_chn, dim_size, stride)]
            self.in_chn = dim_size * 1
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        c1 = self.activation(self.bn1(self.conv1(x)))
        c1 = self.activation(self.bn2(self.conv2(c1)))

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.lateral3(c4))
        p3 = self._upsample_add(p4, self.lateral2(c3))

        return p3

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

class MultiSensor3D(nn.Module):
    def __init__(self):
        super(MultiSensor3D, self).__init__()

    resnet18 = models.resnet18(pretrained=True)
