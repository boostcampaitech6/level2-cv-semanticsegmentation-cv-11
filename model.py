from torchvision.models.segmentation import (
    fcn_resnet50, 
    fcn_resnet101, 
    lraspp_mobilenet_v3_large, 
    deeplabv3_mobilenet_v3_large, 
    deeplabv3_resnet50, 
    deeplabv3_resnet101,
)
from torchvision.models.segmentation import FCN_ResNet50_Weights 
from monai.networks.nets import (
    UNet,
    UNETR,
    SwinUNETR,
)

import torch
import torch.nn as nn
from collections.abc import Sequence
from monai.networks.layers.factories import Act, Norm

class MyUNet(nn.Module):
    def __init__(self, spatial_dims=2, in_channels=3, out_channels=29, channels=(4, 8, 16, 32, 64), strides=(2, 2, 2, 2)):
        super(MyUNet, self).__init__()
        self.model = UNet(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, channels=channels, strides=strides)
    
    def forward(self, x):
        x = self.model(x)
        return x

class MyUNETR(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=29, img_size=512):
        super(MyUNETR, self).__init__()
        self.model = UNETR(in_channels=in_channels, out_channels=out_channels, img_size=img_size)
    
    def forward(self, x):
        x = self.model(x)
        return x

class Myfcn_resnet50(nn.Module):
	def __init__(self):
		super(Myfcn_resnet50, self).__init__()
		self.model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
		# output class 개수를 dataset에 맞도록 수정합니다.
		self.model.classifier[4] = nn.Conv2d(512, 29, kernel_size=1)
	
	def forward(self, x):
		x = self.model(x)
		return x

# 사용 가능한 손실 함수의 진입점
_model_entrypoints = {
    "unet": MyUNet,
    "unetr": MyUNETR,
    "swinunetr": SwinUNETR,
    "fcn_resnet50": Myfcn_resnet50,
    "fcn_resnet101": fcn_resnet101,
}

def model_entrypoint(model_name):
    return _model_entrypoints[model_name]

def is_model(model_name):
    return model_name in _model_entrypoints

def create_model(model_name):
    
    if is_model(model_name):
        _create_model = model_entrypoint(model_name)
        model = _create_model()
    else:
        raise RuntimeError("Unknown model (%s)" % model_name)
    
    return model