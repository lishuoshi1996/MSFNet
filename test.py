from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from net import *
import torch.nn as nn
import torch.nn.functional as F
import torch


image_name = './test.png'
cuda = torch.cuda.is_available()
cuda = False
model = final_Net()
# Load pretrained models
model.load_state_dict(torch.load("./pretrained/indoor.pth" )["state_dict"])
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

image = Image.open(image_name)
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
    im = Image.fromarray(prediction)
    im.save('./result.png')