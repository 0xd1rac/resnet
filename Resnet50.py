from Resnet import Resnet
class Resnet50():
    def __init__(self, 
                img_channels:int=3, 
                num_classes:int=1000
                ):
        
        self.model = Resnet([3,4,6,3], img_channels, num_classes)