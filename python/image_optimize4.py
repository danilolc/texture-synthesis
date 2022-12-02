#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[0]: Load everything
import torch
import scipy.ndimage as nd
from torch import optim, nn
from torchvision import models, transforms, utils
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt

#######################

vgg19 = models.vgg19(weights = "DEFAULT").to("cuda")
#vgg19 = models.vgg19().to("cuda")

for i in range(len(vgg19.features)):
    #if type(vgg19.features[i]) is nn.modules.conv.Conv2d:
    #    vgg19.features[i].paddin_gmode = 'reflect'
        #vgg19.features[i].padding=(0,0)

    if type(vgg19.features[i]) is nn.modules.pooling.MaxPool2d:
        vgg19.features[i] = nn.AvgPool2d(2)
        
    #if type(vgg19.features[i]) is torch.nn.modules.activation.ReLU:
    #    vgg19.features[i] = nn.LeakyReLU(0.001, inplace=True)

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
    #return transforms.ToPILImage()(image)
    return image.cpu().detach().permute(1, 2, 0)

#######################
    
def gram(k):
    N, W, H = k.size()
    f = torch.flatten(k, 1)
    g = torch.matmul(f, f.transpose(0, 1))
    return g * (10000 / (2*N*W*H))
    
# In[1]:


    












# In[1]: Lê imagem e plota

im = Image.open("textures/sky6.jpg").convert('RGB')
im = nd.zoom(im, (1/2,1/2,1))
#im = Image.open("tests/electric2.jpg").convert('RGB')
im = preprocess(im).to("cuda")

#plt.imshow(deprocess(im))
plt.imshow(im.cpu().detach().permute(1, 2, 0))
plt.show()

c1 = C1(im).detach()
c2 = C2(c1).detach()
c3 = C3(c2).detach()
c4 = C4(c3).detach()
c5 = C5(c4).detach()

g1 = gram(c1)
g2 = gram(c2)
g3 = gram(c3)
g4 = gram(c4)
g5 = gram(c5)

del im, c1, c2, c3, c4, c5

# In[]: Lê conteúdo e plota 

im = Image.open("textures/style/eu.jpg").convert('RGB')
im = nd.zoom(im, (2,2,1))

im = preprocess(im).to("cuda")

plt.imshow(im.cpu().detach().permute(1, 2, 0))
plt.show()

k1 = C1(im).detach()
k2 = C2(k1).detach()
k3 = C3(k2).detach()
k4 = C4(k3).detach()
k5 = C5(k4).detach()

# In[]:











# In[]: Prepair to train

#Tensor = torch.cuda.FloatTensor
#image = Variable(Tensor(image), requires_grad=True)
#image = image.requires_grad_()

image = torch.rand(im.size(), requires_grad=True, device="cuda")

loss_func = nn.MSELoss()
optimizer = optim.LBFGS([image], lr=1e-0, history_size=50)

pbar = tqdm(range(1201))
def closure():
    optimizer.zero_grad()
    
    ######################## 1
    _c1 = C1(image)
    content_loss = loss_func(_c1, k1)
    
    _g1 = gram(_c1)
    style_loss = loss_func(_g1, g1)
    
    
    ######################## 2
    _c2 = C2(_c1)
    content_loss = loss_func(_c2, k2) + content_loss
    
    _g2 = gram(_c2)
    style_loss = loss_func(_g2, g2) + style_loss


    ######################## 3
    _c3 = C3(_c2)
    content_loss = loss_func(_c3, k3) + content_loss
    
    _g3 = gram(_c3)
    style_loss = loss_func(_g3, g3) + style_loss


    ######################## 4
    _c4 = C4(_c3)
    content_loss = loss_func(_c4, k4) + content_loss
    
    _g4 = gram(_c4)
    style_loss = loss_func(_g4, g4) + style_loss


    ######################## 5
    #_c5 = C5(_c4)
    #content_loss = loss_func(_c5, k5) + content_loss
    
    #_g5 = gram(_c5)    
    #style_loss = loss_func(_g5, g5) + style_loss
    
    
    ######################## L
    loss = style_loss + 2 * content_loss
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
    if i % 50 == 0:
        utils.save_image(deprocess(image).permute(2, 0, 1), f"clouds/{i}.png")

        optimizer = optim.LBFGS([image], history_size=50)
        with torch.no_grad():
            image.add_(torch.randn(image.size(), device="cuda") * 1)
            #image.clamp_(0, 1)

    optimizer.step(closure)
    plt.imshow(deprocess(image))
    plt.show()
    


