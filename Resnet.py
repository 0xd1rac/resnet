from ResnetBlock import ResnetBlock
import torch
import torch.nn as nn
from typing import List

class Resnet(nn.Module):
    def __init__(self,
                 layers: List[int],
                 image_channels: int,
                 num_classes: int
                 ):
        super().__init__()
        self.in_channels = 64

        self.conv_1 = nn.Conv2d(image_channels, 
                                64, 
                                kernel_size=7,
                                stride=2,
                                padding=3
                                )
        
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1
                                    )

        # Resnet layers
        self.layer_1 = self._make_layer(layers[0], out_channels=64, stride=1)
        self.layer_2 = self._make_layer(layers[1], out_channels=128, stride=2)
        self.layer_3 = self._make_layer(layers[2], out_channels=256, stride=2)
        self.layer_4 = self._make_layer(layers[3], out_channels=512, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d()
        self.fc = nn.Linear(512*4, num_classes)

    
    def _make_layer(self, 
                    num_resnet_blocks:int, 
                    out_channels:int,
                    stride:int
                    ):
        identity_downsample = None
        resnet_blocks = list()
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                                    nn.Conv2d(self.in_channels, 
                                              out_channels*4, 
                                              kernel_size=1, 
                                              stride=stride
                                              ),
                                    nn.BatchNorm2d(out_channels * 4)
                                )

        resnet_blocks.append(ResnetBlock(self.in_channels,out_channels,identity_downsample,stride))
        self.in_channels = out_channels * 4 

        for i in range(num_resnet_blocks - 1):
            resnet_blocks.append(ResnetBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*resnet_blocks)
        

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn_1(self.conv_1(x))))
        x = self.layer_4(self.layer_3(self.layer_2(self.layer_1(x))))
        x = self.avg_pool(x)
        x  = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x





