from ResNet import ResNet
class Resnet101():
    def __init__(self, 
                img_channels:int=3, 
                num_classes:int=1000
                ):
        
        self.model = ResNet([3,4,23,3], img_channels, num_classes)