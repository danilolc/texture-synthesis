#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from model import convs


#### Calculate the Gram Matrix from features tensor k


def gram(k):
    N, W, H = k.size()
    f = torch.flatten(k, 1)
    g = torch.matmul(f, f.transpose(0, 1))
    return g * (10000 / (2*N*W*H))


#### Calculate all Gram Matrix representation from image 


def gram_vec(im):
    
    gv = []
    
    for C in convs:
        im = C(im).detach()
        gv.append( gram(im) )
        
    return gv