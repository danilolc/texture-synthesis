#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as nd

normalize = False

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

imean = [-2.1179, -2.0357, -1.8044]
istd  = [ 4.3668,  4.4643,  4.4444]

def preprocess(image):
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean, std)(image)
    return image.to('cuda')

def deprocess(image):
    if normalize:
        image = transforms.Normalize(imean, istd)(image)
    return image.cpu().detach().permute(1, 2, 0)

def load_image(pth, scale=1):
    im = Image.open(pth).convert('RGB')
    im = nd.zoom(im, (scale, scale, 1))
    return preprocess(im).to("cuda")
    
def plot_image(tensor):
    plt.imshow(deprocess(tensor))
    plt.show()