import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)



class VGG19_BN(nn.Module):
    '''
    VGG network
    features: model layers of feature extraction
    num_classes = 10  (MNIST)
    '''
    def __init__(self, num_classes=10, init_weights=True):
        super(VGG19_BN, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
 
        # feature extraction
        self.features = self.make_layers(self.cfg, batch_norm=True)
 
        # adaptive average pooling, feature maps pooled to 7x7
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
 
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),   # 512*7*7 --> 4096
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),          # 4096 --> 4096
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),   # 4096 --> 10
        )
       
        # weight initialization
        if init_weights:
            self._initialize_weights()
 
    def make_layers(self, cfg, batch_norm=True):
        layers = []
 
        in_channels = 1
 
        # traverse cfg
        for v in cfg:
            if v == 'M':    # add maxpooling
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
           
            else:           # add conv
                # 3×3 conv
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
 
                # conv --> BN --> ReLU
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
 
                # input_channels of next layer = current output_channels
                in_channels = v
 
        # return model layers in sequential format
        return nn.Sequential(*layers)
 
    def forward(self, x):
 
        out = self.features(x)
        out = self.avgpool(out)
 
        # feature maps flatten to vector
        out = torch.flatten(out, 1)
 
        out = self.classifier(out)
        # out = F.softmax(out, dim=1)
 
        return out
 
    def _initialize_weights(self):
        '''
        weight initialization
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # conv layers using kaiming initialization
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                # bias initialized to 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # BN initialized to 1
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # fc layer initialization
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)