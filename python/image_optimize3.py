#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[0]: Load everything
import torch
from torch import optim, nn
from torchvision import models, transforms
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt

#######################

vgg19 = models.vgg19(weights = "DEFAULT").to("cuda")
#vgg19 = models.vgg19().to("cuda")

for i in range(len(vgg19.features)):
    if type(vgg19.features[i]) is nn.modules.conv.Conv2d:
        vgg19.features[i].padding_mode = 'reflect'
        vgg19.features[i].padding=(0,0)

    if type(vgg19.features[i]) is nn.modules.pooling.MaxPool2d:
        vgg19.features[i] = nn.AvgPool2d(2)

C1 = vgg19.features[0:4]
C2 = vgg19.features[4:9]
C3 = vgg19.features[9:18]
C4 = vgg19.features[18:27]
C5 = vgg19.features[27:36]

del vgg19

#######################

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

#######################
    
def gram(k):
    N, W, H = k.size()
    f = torch.flatten(k, 1)
    g = torch.matmul(f, f.transpose(0, 1))
    return g * (10000 / (2*N*W*H))


def gram_vec(im):
    
    cim_1 = C1(im).detach()
    gim_1 = gram(cim_1)
    
    cim_2 = C2(cim_1).detach()
    gim_2 = gram(cim_2)
    
    cim_3 = C3(cim_2).detach()
    gim_3 = gram(cim_3)
    
    cim_4 = C4(cim_3).detach()
    gim_4 = gram(cim_4)
    
    cim_5 = C5(cim_4).detach()
    gim_5 = gram(cim_5)
    
    return [gim_1, gim_2, gim_3, gim_4, gim_5]

    
# In[1]:


    












# In[1]: Lê imagem e plota

im = Image.open("textures/earth.png").convert('RGB')
im = preprocess(im).to("cuda")

#plt.imshow(deprocess(im))
plt.imshow(im.cpu().detach().permute(1, 2, 0))
plt.show()

gim = gram_vec(im)
#del im

# In[]: Generate mean gram matrix

names = [f"mean/clouds{i+1}.jpg" for i in range(8)]
#names = ["vg.jpg", 'vg2.jpg']
    
M = []
for n in names:
    im = Image.open(n).convert('RGB')
    g = gram_vec(preprocess(im).to("cuda"))
    M.append(g)
gim = []
for i in range(5):
    v = torch.stack([M[0][i], M[1][i], M[2][i], M[3][i], M[4][i], M[5][i], M[6][i], M[7][i]])
    g = torch.mean(v, 0)
    gim.append(g)
    
# In[]:















# In[]: Prepair to train

#Tensor = torch.cuda.FloatTensor
#image = Variable(Tensor(image), requires_grad=True)    
#image = image.requires_grad_()


image = im.requires_grad_()
#image = torch.rand(3, 256, 256, requires_grad=True, device="cuda")
#image = torch.zeros(3, 256, 256, requires_grad=True, device="cuda")

loss_func = nn.MSELoss()
optimizer = optim.LBFGS([image], history_size=50)

pbar = tqdm(range(601))
def closure():
    optimizer.zero_grad()
    
    c_1 = C1(image)
    g_1 = gram(c_1)
    
    c_2 = C2(c_1)
    g_2 = gram(c_2)

    c_3 = C3(c_2)
    g_3 = gram(c_3)

    c_4 = C4(c_3)
    g_4 = gram(c_4)

    #c_5 = C5(c_4)
    #g_5 = gram(c_5)
    
    loss =        loss_func(g_1, gim[0])
    loss = loss + loss_func(g_2, gim[1])
    loss = loss + loss_func(g_3, gim[2])
    loss = loss + loss_func(g_4, gim[3])
    #loss = loss + loss_func(g_5, gim[4])
    
    loss.backward()
    
    pbar.set_description("%.8f" % loss)

    return loss


# In[]: Train image
for i in pbar:
    optimizer.step(closure)

    plt.imshow(deprocess(image))
    plt.show()

# In[]: Train adding noise
    
# Turn off normalization
for i in pbar:
    if i % 30 == 0:
        plt.imshow(deprocess(image))
        plt.show()
        
        optimizer = optim.LBFGS([image], history_size=50)
        with torch.no_grad():
            image.add_(torch.randn(image.size(), device="cuda") * 1)
            image.clamp_(0, 1)

    optimizer.step(closure)
    
# In[]: Train translating

spd = 25

# Turn off normalization
for i in pbar:
    if i % 30 == 0:
        plt.imshow(deprocess(image))
        plt.show()
        
        optimizer = optim.LBFGS([image], history_size=50)
        with torch.no_grad():
            r = torch.rand(3, 25, 512, device="cuda")
            i = torch.cat([image[:,spd:], r], dim=1) 
            image.add_(-image)
            image.add_(i)

    optimizer.step(closure)

