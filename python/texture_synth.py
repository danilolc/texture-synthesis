#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from tqdm import tqdm

from image import load_image, plot_image
from gram import gram, gram_vec
from model import device, convs, convs_vec
from torchvision.utils import save_image

#### Load source image

# Image used as sample or as style
src = load_image("textures/128/sky.png").to(device)
gsrc = gram_vec(src)

plot_image(src)


#### Load content image

# Turn on if you want to do Style Transfer
has_content = False

if has_content:
    # Content image from Style Transfer
    con = load_image("textures/new/van-gogh.jpg").to(device)
    ccon = convs_vec(con)
    
    plot_image(con)


##### Create random noise, loss function and optimizer

# Size of the output from Texture Synthesis
dst_size = (3, 256, 256)

if has_content:
    dst_size = con.size()

dst = torch.rand(dst_size, requires_grad=True, device=device)

loss_func = nn.MSELoss()
optimizer = optim.LBFGS([dst], history_size=50)


##### Closure function that calculate the loss


# Features weights on each layer of the network
wcontent = [0, 0, 3, 3, 3]
wstyle = [1, 1, 3, 1, 1]

pbar = tqdm(range(501))
def closure():
    optimizer.zero_grad()
    
    loss = torch.tensor(0, device=device)
    cdst = dst
    
    for i, C in enumerate(convs):
        cdst = C(cdst)
        if wstyle[i] != 0:
            gdst = gram(cdst)
            loss = loss + loss_func(gdst, gsrc[i]) * wstyle[i]
        if has_content and wcontent[i] != 0:
            loss = loss + 1 * loss_func(cdst, ccon[i]) * wcontent[i]
    
    loss.backward()
    
    pbar.set_description("%.8f" % loss)

    return loss


##### Run optimization

# Turn on if you want to add normal noise to the image each 25 iterations
# It will save the result image before adding the noise
add_noise = False

for i in pbar:
    
    if add_noise and i % 25 == 0:
        save_image(dst, f"{i}.png")
        optimizer = optim.LBFGS([dst], history_size=50) # Reset optimizer
        with torch.no_grad():
            dst.add_(torch.randn(dst.size(), device=device) * 0.7)
            dst.clamp_(0, 1)

    optimizer.step(closure)
    plot_image(dst)

