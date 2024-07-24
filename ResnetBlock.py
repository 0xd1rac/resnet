import torch
import torch.nn as nn 

class ResnetBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 identity_downsample: bool=None,
                 stride: int=1
                 ):
        
        super().__init__()
        self.expansion = 4 
    
        self.conv_1 = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0
                                )
        self.bn_1 = nn.BatchNorm2d(out_channels)

        self.conv_2 = nn.Conv2d(out_channels,
                                out_channels,
                                kernel_size=3,
                                stride=stride,
                                padding=1
                                )
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.conv_3 = nn.Conv2d(out_channels,
                                out_channels * self.expansion,
                                kernel_size=1,
                                stride=1,
                                padding=0
                                )
        self.bn_3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    

    def forward(self, x):
        identity = x 

        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))
        x = self.bn_3(self.conv_3(x))

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x
        