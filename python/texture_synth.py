#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from tqdm import tqdm

from image import load_image, plot_image
from gram import gram, gram_vec
from model import convs, convs_vec


#### Load source image


src = load_image("textures/128/onca.jpg")
gsrc = gram_vec(src)

plot_image(src)


#### Load content image


has_content = False

if has_content:
    con = load_image("textures/style/eu.jpg")
    ccon = convs_vec(con)
    
    plot_image(con)


##### Create random noise, loss function and optimizer


dst_size = (3, 256, 256)

if has_content:
    dst_size = con.size()

dst = torch.rand(dst_size, requires_grad=True, device="cuda")

loss_func = nn.MSELoss()
optimizer = optim.LBFGS([dst], history_size=50)


##### Closure function that calculate the loss


pbar = tqdm(range(601))
def closure():
    optimizer.zero_grad()
    
    loss = torch.tensor(0, device='cuda')
    cdst = dst
    
    for i, C in enumerate(convs):
        cdst = C(cdst)
        gdst = gram(cdst)
        loss = loss + loss_func(gdst, gsrc[i])
        if has_content:
            loss = loss + 2 * loss_func(cdst, ccon[i])
    
    loss.backward()
    
    pbar.set_description("%.8f" % loss)

    return loss


##### Run optimization


add_noise = False

for i in pbar:
    
    if add_noise and i % 50 == 0:
        optimizer = optim.LBFGS([dst], history_size=50) # Reset optimizer
        with torch.no_grad():
            dst.add_(torch.randn(dst.size(), device="cuda") * 1)

    optimizer.step(closure)
    plot_image(dst)

