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

cuda = torch.cuda.is_available()
cuda = False
model = final_Net()
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

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

for filename in filepath:
    image = Image.open(image_path + '/'+filename)
    if image.mode != "RGB":
        image = to_rgb(image)
    image = transform(image).unsqueeze_(0)
    model.eval()  # Load model
    input = Variable(image.type(Tensor))
    with torch.no_grad():
        _,_,output = model(input)
        output = torch.clamp(output, 0., 1.)
        prediction = output.data.cpu().numpy().squeeze().transpose((1,2,0))
        prediction = (prediction*255.0).astype("uint8")
    label_name = filename.split('_')[0]
    try:
        label = Image.open(label_path +'/'+label_name+'.png')
    except:
        label = Image.open(label_path +'/'+label_name+'.jpg')
    label = np.array(label)
    psnr = measure.compare_psnr(prediction, label, 255)
    ssim = measure.compare_ssim(prediction, label, data_range=255, multichannel=True)
    psnrs.append(psnr)
    ssims.append(ssim)
    print(len(psnrs))
    print(psnr)
psnr_mean = sum(psnrs) / len(psnrs)
ssim_mean = sum(ssims) / len(ssims)
print('PSNR:%f,SSIM:%f'%(psnr_mean,ssim_mean))

psnr_mean,ssim_mean = to_psnr_ssim(image_path,label_path)
print(tag+model+':PSNR:%f,SSIM:%f'%(psnr_mean,ssim_mean))