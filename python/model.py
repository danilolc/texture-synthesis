#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
from torchvision import models

device = "cuda"
#device = "cpu"

vgg19 = models.vgg19(weights = "DEFAULT").to(device)

#### Adapt VGG19 to texture synthesis


for i in range(len(vgg19.features)):
    if type(vgg19.features[i]) is nn.modules.conv.Conv2d:
        vgg19.features[i].padding_mode = 'reflect'
        #vgg19.features[i].padding=(0,0)

    if type(vgg19.features[i]) is nn.modules.pooling.MaxPool2d:
        vgg19.features[i] = nn.AvgPool2d(2)


#### Load each piece of the network


"""
C1 = vgg19.features[0:4]
C2 = vgg19.features[4:9]
C3 = vgg19.features[9:18]
C4 = vgg19.features[18:27]
C5 = vgg19.features[27:36]
"""

#"""

C1 = vgg19.features[0:2]
C2 = vgg19.features[2:5]
C3 = vgg19.features[5:10]
C4 = vgg19.features[10:19]
C5 = vgg19.features[19:28]
C6 = vgg19.features[28:36]
#"""

convs = [C1, C2, C3, C4, C5]

def convs_vec(im):    
    cv = []
    
    for C in convs:
        im = C(im).detach()
        cv.append(im)
        
    return cv

del vgg19
