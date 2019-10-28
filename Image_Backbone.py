import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision

class ImageBackbone(nn.Module):

	output1 = None
	output2 = None
	output3 = None

	def __init__(self, in_chn, out_chn=1, as_backbone=False, pretrained=False):

		super(ImageBackbone, self).__init__()

		self._in_chn = in_chn
		self._out_chn = out_chn
		self._as_backbone = as_backbone
		self._pretrained = pretrained

		resnet = torchvision.models.resnet18(pretrained=self._pretrained)

		#Remove adaptiveavgpool & fc layer
		#Output will be 512 chn at 7/224 of original input
		self._model = nn.Sequential(*list(resnet.children())[:-2])

		#Modify in_chn to fit input
		self._model[0] = nn.Conv2d(self._in_chn, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

		#Get ResNet-18 intermediate layer feature maps
		if self._as_backbone:
			self._model[4].register_forward_hook(self.hook_layer1) #resnet_18.layer1
			self._model[5].register_forward_hook(self.hook_layer2) #resnet_18.layer2
			self._model[6].register_forward_hook(self.hook_layer3) #resnet_18.layer3

		#four conv + 2 bilinear upsample for dense depth
		self._conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
		self._conv2 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
		self._conv3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
		self._conv4 = nn.Conv2d(1024, self._out_chn, kernel_size=1, stride=1) #1x1 Conv for a single-channel depth map

	def forward(self, x):
		#output4 will be 7/224 of input
		output4 = self._model(x)

		#output5 will be 56/224 (1/4) of input
		output5 = f.interpolate(output4, scale_factor=4, mode='bilinear')
		output5 = f.interpolate(output5, scale_factor=2, mode='bilinear')

		#out will be 28/224 (1/8) of input
		out = self._conv1(output5)
		out = self._conv2(out)
		out = self._conv3(out)
		out = self._conv4(out)

		#out will be 224/224 (1/1) of input
		out = f.interpolate(out, scale_factor=4, mode='bilinear')
		out = f.interpolate(out, scale_factor=2, mode='bilinear')

		global output1
		global output2
		global output3

		if self._as_backbone:
			return out, output1, output2, output3, output4, output5

		else:
			return out

	def hook_layer1(self, module, input, output):
		global output1
		output1 = output

	def hook_layer2(self, module, input, output):
		global output2
		output2 = output

	def hook_layer3(self, module, input, output):
		global output3
		output3 = output



