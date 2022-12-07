#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as nd


def preprocess(image):
    image = transforms.ToTensor()(image)
    return image.to('cuda')

def deprocess(image):
    return image.cpu().detach().permute(1, 2, 0)


def load_image(pth, scale=1):
    im = Image.open(pth).convert('RGB')
    im = nd.zoom(im, (scale, scale, 1))
    return preprocess(im).to("cuda")
    
def plot_image(tensor):
    plt.imshow(deprocess(tensor).clamp(0,1))
    plt.show()