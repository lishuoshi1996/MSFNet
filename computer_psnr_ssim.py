import cv2
import os
import numpy as np
from skimage import measure

tag = 'indoor'
image_path = './SOTS/'+tag +'/hazy'
label_path = './SOTS/'+tag +'/clear'

filepath = []
for root_1,ds,fs in os.walk(image_path):
    for f in fs:
        if f[-3:] == "jpg" or f[-3:] == "png":
            filepath.append(f)
psnrs = []
ssims = []
print(len(filepath))
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from net import *
import torch.nn as nn
import torch.nn.functional as F
import torch
model = final_Net()
cuda = torch.cuda.is_available()
# Load pretrained models
model.load_state_dict(torch.load("./pretrained/"+tag+".pth" )["state_dict"])
if cuda:
    model = model.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [
    transforms.ToTensor(), 
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
]
transform = transforms.Compose(transforms_)
transforms_gt = [
    transforms.ToTensor()
]
transforms_gt = transforms.Compose(transforms_gt)

for filename in filepath:
    image = Image.open(image_path + '/'+filename)
    image = transform(image).unsqueeze_(0)
    model.eval()  # Load model
    input = Variable(image.type(Tensor))
    with torch.no_grad():
        _,_,output = model(input)
        output = torch.clamp(output, 0., 1.)
        prediction = output.data.cpu().numpy().squeeze().transpose((1,2,0))
    label_name = filename.split('_')[0]
    try:
        label = Image.open(label_path +'/'+label_name+'.png')
    except:
        label = Image.open(label_path +'/'+label_name+'.jpg')
    label = transforms_gt(label)
    label = label.data.cpu().numpy().transpose((1,2,0))
    psnr = measure.compare_psnr(prediction, label, 1)
    ssim = measure.compare_ssim(prediction, label, data_range=1, multichannel=True)
    psnrs.append(psnr)
    ssims.append(ssim)
psnr_mean = sum(psnrs) / len(psnrs)
ssim_mean = sum(ssims) / len(ssims)
print('PSNR:%f,SSIM:%f'%(psnr_mean,ssim_mean))
