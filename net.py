# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from residual_dense_block import RDB

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(channel_in, channel_out, stride)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out

class better_upsampling(nn.Module):
      def __init__(self, in_ch, out_ch):
          super(better_upsampling, self).__init__()
          self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=0)

      def forward(self, x,y):
          x = nn.functional.interpolate(x,size= y.size()[2:], mode='nearest', align_corners=None)
          x = F.pad(x, (3 // 2, int(3 / 2), 3 // 2, int(3 / 2)))
          x = self.conv(x)
          return x

class down_Block(nn.Module):
    def __init__(self, in_channels, stride=2):
        kernel_size=3
        super(down_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, stride*in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out
        
class low_level_fearture_fusdion_Block(nn.Module):
    def __init__(self,attention=True):
        super(low_level_fearture_fusdion_Block, self).__init__()
        self.up_medium = better_upsampling(32,16)
        self.up_high = better_upsampling(64,16)
        self.refine = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((5,16))), requires_grad=attention)

    def forward(self,x,y,z):
        y_up = self.up_medium(y,x)
        z_up = self.up_high(z,x)
        add = self.coefficient[0, :][None, :, None, None] *x + self.coefficient[ 1, :][None, :, None, None] *y_up + \
            self.coefficient[ 2, :][None, :, None, None] *z_up
        refine_feature = self.refine(add)
        out = self.coefficient[ 3, :][None, :, None, None] *x + self.coefficient[ 4, :][None, :, None, None] *refine_feature
        return out

class medium_level_fearture_fusdion_Block(nn.Module):
    def __init__(self,attention=True):
        super(medium_level_fearture_fusdion_Block, self).__init__()
        self.down_low = down_Block(16)
        self.up_high = better_upsampling(64,32)
        self.refine = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((5,32))), requires_grad=attention)

    def forward(self,x,y,z):
        x_down = self.down_low(x)
        z_up = self.up_high(z,y)
        add = self.coefficient[0, :][None, :, None, None] *x_down + self.coefficient[ 1, :][None, :, None, None] *y + \
            self.coefficient[ 2, :][None, :, None, None] *z_up
        refine_feature = self.refine(add)
        out = self.coefficient[ 3, :][None, :, None, None] *y + self.coefficient[ 4, :][None, :, None, None] *refine_feature
        return out

class high_level_fearture_fusdion_Block(nn.Module):
    def __init__(self,attention=True):
        super(high_level_fearture_fusdion_Block, self).__init__()
        self.down_low_1 = down_Block(16)
        self.down_low_2 = down_Block(32)
        self.down_medium =  down_Block(32)
        self.refine = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.coefficient = nn.Parameter(torch.Tensor(np.ones((5,64))), requires_grad=attention)

    def forward(self,x,y,z):
        x_down = self.down_low_1(x)
        x_down = self.down_low_2(x_down)
        y_down = self.down_medium (y)
        add = self.coefficient[0, :][None, :, None, None] *x_down + self.coefficient[ 1, :][None, :, None, None] *y_down + \
            self.coefficient[ 2, :][None, :, None, None] *z
        refine_feature = self.refine(add)
        out = self.coefficient[ 3, :][None, :, None, None] *z + self.coefficient[ 4, :][None, :, None, None] *refine_feature
        return out

class fearture_fusdion_Block(nn.Module):
    def __init__(self,attention=True):
        super(fearture_fusdion_Block, self).__init__()
        self.low_feature_fusion = low_level_fearture_fusdion_Block()
        self.medium_feature_fusion = medium_level_fearture_fusdion_Block()
        self.high_feature_fusion = high_level_fearture_fusdion_Block()
        self.conv1 = RDB(16,4,16)
        self.conv2 = RDB(32,4,16)
        self.conv3 = RDB(64,4,16)

    def forward(self,x,y,z):
        x_ff = self.low_feature_fusion(x,y,z)
        y_ff = self.medium_feature_fusion(x,y,z)
        z_ff = self.high_feature_fusion(x,y,z)
        x_out = self.conv1(x_ff)
        y_out = self.conv2(y_ff)
        z_out = self.conv3(z_ff)
        return (x_out,y_out,z_out)

class first_Net(nn.Module):
    def __init__(self):
        super(first_Net, self).__init__()

        self.conv01 = nn.Conv2d(3, 16, 3, 1, 1)
        self.down_1 = down_Block(16,2)
        self.down_2 = down_Block(32,2)

        self.conv1 = RDB(16,4,16)
        self.conv2 = RDB(32,4,16)
        self.conv3 = RDB(64,4,16)

        self.f1 = fearture_fusdion_Block()
        self.f2 = fearture_fusdion_Block()
        
        
    def forward(self, x):
        
        x0 =  self.conv01(x)
        y0 =  self.down_1(x0)
        z0 =  self.down_2(y0)

        x0 =  self.conv1(x0)
        y0 =  self.conv2(y0)
        z0 =  self.conv3(z0)
        x,y,z = self.f1(x0,y0,z0)
        x,y,z = self.f2(x,y,z)
        return (x,y,z)

class second_Net(nn.Module):
    def __init__(self, in_c=3, out_c=3, only_residual=True):
        super(second_Net, self).__init__()

        self.conv11= RDB(64,4,16)
        self.conv12 = RDB(64,4,16)
        self.conv13 = BasicBlock(64,64)
        self.conv14 = BasicBlock(64,3)
        self.up1 = better_upsampling(64,32)
        self.gate21 = BasicBlock(64,32)
        self.conv21 = RDB(32,4,16)
        self.conv22 = RDB(32,4,16)
        self.conv23 = BasicBlock(32,32)
        self.conv24 = BasicBlock(32,3)
        self.up2 = better_upsampling(32,16)
        self.gate31 = BasicBlock(32,16)
        self.conv31 = RDB(16,4,16)
        self.conv32 = RDB(16,4,16)
        self.conv41 = BasicBlock(16,16)
        self.conv42 = BasicBlock(16,3)
    def forward(self, z,y,x):
        x11 = self.conv11(x)
        x11 = self.conv12(x11)
        x_out = self.conv13(x11)
        x_out = self.conv14(x_out)
        x11_2 = self.up1(x11,y)
        y_1 = self.gate21(torch.cat((x11_2,y),1))
        y_2 = self.conv21(y_1)
        y_3 = self.conv22(y_2)
        y_4 = self.conv23(y_3)
        y_out = self.conv24(y_4)
        y_up = self.up2(y_3,z) 
        z_1 =  self.gate31(torch.cat((y_up,z),1))
        z_2 = self.conv31(z_1)
        z_3 = self.conv32(z_2)
        z_4 = self.conv41(z_3)
        z_out = self.conv42(z_4)
        return (x_out,y_out,z_out)


class final_Net(nn.Module):
    def __init__(self):
        super(final_Net, self).__init__()
        self.first = first_Net()
        self.second = second_Net()
    def forward(self, x):
        x,y,z = self.first(x)
        x,y,z = self.second(x,y,z)
        return (x,y,z)
