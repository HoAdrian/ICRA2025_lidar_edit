#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D




class BEV_Unet(nn.Module):

    def __init__(self,n_class,n_height,dilation = 1,group_conv=False,input_batch_norm = False,dropout = 0.,circular_padding = False, dropblock = True, use_vis_fea=False):
        super(BEV_Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        if use_vis_fea:
            self.network = UNet(n_class*n_height,2*n_height,dilation,group_conv,input_batch_norm,dropout,circular_padding,dropblock)
        else:
            self.network = UNet(n_class*n_height,n_height,dilation,group_conv,input_batch_norm,dropout,circular_padding,dropblock)

    def forward(self, x):
        #print("++++ BEV UNET")
        x = self.network(x)
        #print(">> x: ", x.shape)
        x = x.permute(0,2,3,1) #(B,H,W,n_class*n_height)
        #print(">> permuted x: ", x.shape)
        new_shape = list(x.size())[:3] + [self.n_height,self.n_class]
        x = x.view(new_shape)
        #print(">>reshaped x: ", x.shape)
        x = x.permute(0,4,1,2,3)
        #print(">>final x: ", x.shape)
        return x
    
class UNet(nn.Module):
    '''
    UNet architecture. 

    n_height: in_channel
    n_class: out_channel
    Input: (B, n_height, H, W)
    Output: (B, n_class, H, W)

    Assuming H and W are divisible by 8, otherwise the shape would be a bit different (rounded down when not divisible)
    x2 = down1(x1): (B, 64, H, W) ===> (B, 128, H//2, W//2)              
    x3 = down2(x2): (B, 128, H//2, W//2) ===> (B, _256 H//4, W//4)
    x4 = down3(x3): (B, 256, H//4, W//4) ===> (B, 512, H//8, W//8)
    x5 = down4(x4): (B, 512, H//8, W//8) ===> (B, 512, H//16, W//16)

    x = up1(x5, x4): (B, 512, H//16, W//16) ===> (B, 256, H//8, W//8)
    x = up2(x, x3): (B, 256, H//8, W//8) ===> (B, 128, H//4, W//4)
    x = up3(x, x2): (B, 128, H//4, W//4) ===> (B, 64, H//2, W//2)
    x = up4(x, x1): (B, 64, H//2, W//2) ===> (B, 64, H, W)


    '''
    def __init__(self, n_class,n_height,dilation,group_conv,input_batch_norm, dropout,circular_padding,dropblock):
        super(UNet, self).__init__()
        self.inc = inconv(n_height, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)
        self.up1 = up(1024, 256, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up2 = up(512, 128, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up3 = up(256, 64, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up4 = up(128, 64, circular_padding, group_conv = group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.dropout = nn.Dropout(p=0. if dropblock else dropout)
        self.outc = outconv(64, n_class)

    def forward(self, x):
        #print("++++ UNET")
        x1 = self.inc(x)
        #print("after in conv: ", x1.shape)
        x2 = self.down1(x1)
        #print("after down1: ", x2.shape)
        x3 = self.down2(x2)
        #print("after down2: ", x3.shape)
        x4 = self.down3(x3)
        #print("after down3: ", x4.shape)
        x5 = self.down4(x4)
        #print("after down4 ", x5.shape)
        x = self.up1(x5, x4)
        #print("after up1: ", x.shape)
        x = self.up2(x, x3)
        #print("after up2s ", x.shape)
        x = self.up3(x, x2)
        #print("after up3 ", x.shape)
        x = self.up4(x, x1)
        #print("after up4 ", x.shape)
        x = self.outc(self.dropout(x))
        #print("after last out conv ", x.shape)
        return x


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1,groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1,groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        #print(f"---- double conv input: {x.shape}")
        x = self.conv(x)
        #print(f"---- double conv after conv: {x.shape}")
        return x

class double_conv_circular(nn.Module):
    '''
    (conv => BN => ReLU) * 2, The height and width are unchanged after convolution here.
    Input: 
    - x: (B, in_ch, H, W)
    Output:
    - x: (B, out_ch, H, W)
    '''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0),groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0),groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        # say x shape: (B, in_chans, H, W)

        #add circular padding
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        #The padding (1, 1, 0, 0) applies:
        # 1 pixel of padding to the left
        # 1 pixel of padding to the right
        # 0 pixels of padding to the top
        # 0 pixels of padding to the bottom
        # x shape: (B, in_chans, H, W+2)

        x = self.conv1(x)
        # H_out = (H + 2*(padding which is 1) - 3)+1 = H
        # W_out = (W + 2 + 2*(padding which is 0) - 3)+1 = W
        

        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv2(x)
        
        return x


class inconv(nn.Module):
    '''
    Height and width are unchanged after convolution, Just the channel is changed. 
    Input: 
    - x: (B, in_ch, H, W)
    Output:
    - x: (B, out_ch, H, W)
    '''
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
            else:
                self.conv = double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)

    def forward(self, x):
        #print(f"### in conv input: {x.shape}")
        x = self.conv(x)
        #print(f"### in conv after conv: {x.shape}")
        return x


class down(nn.Module):
    '''
    Downsampling through max pooling and then convolution
    Input: 
    - x: (B, in_ch, H, W)
    Output:
    - x: (B, out_ch, H//2, W//2)
    '''
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )                

    def forward(self, x):
        #print(f"---- dwon : {x.shape}")
        x = self.mpconv(x)
        #print(f"---- down after mp conv: {x.shape}")
        return x


class up(nn.Module):
    '''
    Upsampling and convolution

    Input: 
    - x1: (B, _, H, W), x2: (B, out_ch, H*2, W*2)
    
    Output:
    - x: (B, out_ch, H*2, W*2)
    '''
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, group_conv=False, use_dropblock = False, drop_p = 0.5):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2,groups = in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch,group_conv = group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch,group_conv = group_conv)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        #print(f"++++ up input: {x1.shape}")
        x1 = self.up(x1) 
        #print(f"++++ up : {x1.shape}")
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2] # difference of H
        diffX = x2.size()[3] - x1.size()[3] # difference of W
        # print(f"++++ up diffY: {diffY}")
        # print(f"++++ up diffX: {diffX}")

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # print(f"++++ x1 pad: {x1.shape}")
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1) #(B, out_chan, H*2, W*2)
        #print(f"++++ x cat: {x.shape}")
        x = self.conv(x)
        #print(f"++++ x conv: {x.shape}")
        if self.use_dropblock:
            x = self.dropblock(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    

if __name__=="__main__":
    B, feat_dim, H, W = 1, 32, 480, 360
    x = torch.ones((B,feat_dim,H,W))
    num_unique_labels = 16
    n_height = 32
    my_BEV_model=BEV_Unet(n_class=num_unique_labels, n_height = n_height, input_batch_norm = True, dropout = 0.5, circular_padding = True, use_vis_fea=False)

    in_ch, out_ch = 32, 64
    dilation = 1

    double_conv_layer = double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)
    double_conv_circular_layer = double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)



    in_conv = inconv(n_height, 64, dilation, input_batch_norm=True, circular_padding=True)


'''
EXAMPLE: 

*** out_data: torch.Size([1, 32, 480, 360])
++++ BEV UNET
++++ UNET
### in conv : torch.Size([1, 32, 480, 360])
---- double conv circular input: torch.Size([1, 32, 480, 360])
---- double conv circular pad: torch.Size([1, 32, 480, 362])
---- double conv circular after conv1: torch.Size([1, 64, 480, 360])
---- double conv circular after conv1 pad: torch.Size([1, 64, 480, 362])
---- double conv circular after conv2: torch.Size([1, 64, 480, 360])
### in conv after conv: torch.Size([1, 64, 480, 360])
after in conv:  torch.Size([1, 64, 480, 360])
---- dwon : torch.Size([1, 64, 480, 360])
---- double conv circular input: torch.Size([1, 64, 240, 180])
---- double conv circular pad: torch.Size([1, 64, 240, 182])
---- double conv circular after conv1: torch.Size([1, 128, 240, 180])
---- double conv circular after conv1 pad: torch.Size([1, 128, 240, 182])
---- double conv circular after conv2: torch.Size([1, 128, 240, 180])
---- down after mp conv: torch.Size([1, 128, 240, 180])
after down1:  torch.Size([1, 128, 240, 180])
---- dwon : torch.Size([1, 128, 240, 180])
---- double conv circular input: torch.Size([1, 128, 120, 90])
---- double conv circular pad: torch.Size([1, 128, 120, 92])
---- double conv circular after conv1: torch.Size([1, 256, 120, 90])
---- double conv circular after conv1 pad: torch.Size([1, 256, 120, 92])
---- double conv circular after conv2: torch.Size([1, 256, 120, 90])
---- down after mp conv: torch.Size([1, 256, 120, 90])
after down2:  torch.Size([1, 256, 120, 90])
---- dwon : torch.Size([1, 256, 120, 90])
---- double conv circular input: torch.Size([1, 256, 60, 45])
---- double conv circular pad: torch.Size([1, 256, 60, 47])
---- double conv circular after conv1: torch.Size([1, 512, 60, 45])
---- double conv circular after conv1 pad: torch.Size([1, 512, 60, 47])
---- double conv circular after conv2: torch.Size([1, 512, 60, 45])
---- down after mp conv: torch.Size([1, 512, 60, 45])
after down3:  torch.Size([1, 512, 60, 45])
---- dwon : torch.Size([1, 512, 60, 45])
---- double conv circular input: torch.Size([1, 512, 30, 22])
---- double conv circular pad: torch.Size([1, 512, 30, 24])
---- double conv circular after conv1: torch.Size([1, 512, 30, 22])
---- double conv circular after conv1 pad: torch.Size([1, 512, 30, 24])
---- double conv circular after conv2: torch.Size([1, 512, 30, 22])
---- down after mp conv: torch.Size([1, 512, 30, 22])
after down4  torch.Size([1, 512, 30, 22])
++++ up input: torch.Size([1, 512, 30, 22])
++++ up : torch.Size([1, 512, 60, 44])
++++ up diffY: 0
++++ up diffX: 1
++++ x1 pad: torch.Size([1, 512, 60, 45])
++++ x cat: torch.Size([1, 1024, 60, 45])
---- double conv circular input: torch.Size([1, 1024, 60, 45])
---- double conv circular pad: torch.Size([1, 1024, 60, 47])
---- double conv circular after conv1: torch.Size([1, 256, 60, 45])
---- double conv circular after conv1 pad: torch.Size([1, 256, 60, 47])
---- double conv circular after conv2: torch.Size([1, 256, 60, 45])
++++ x conv: torch.Size([1, 256, 60, 45])
after up1:  torch.Size([1, 256, 60, 45])
++++ up input: torch.Size([1, 256, 60, 45])
++++ up : torch.Size([1, 256, 120, 90])
++++ up diffY: 0
++++ up diffX: 0
++++ x1 pad: torch.Size([1, 256, 120, 90])
++++ x cat: torch.Size([1, 512, 120, 90])
---- double conv circular input: torch.Size([1, 512, 120, 90])
---- double conv circular pad: torch.Size([1, 512, 120, 92])
---- double conv circular after conv1: torch.Size([1, 128, 120, 90])
---- double conv circular after conv1 pad: torch.Size([1, 128, 120, 92])
---- double conv circular after conv2: torch.Size([1, 128, 120, 90])
++++ x conv: torch.Size([1, 128, 120, 90])
after up2s  torch.Size([1, 128, 120, 90])
++++ up input: torch.Size([1, 128, 120, 90])
++++ up : torch.Size([1, 128, 240, 180])
++++ up diffY: 0
++++ up diffX: 0
++++ x1 pad: torch.Size([1, 128, 240, 180])
++++ x cat: torch.Size([1, 256, 240, 180])
---- double conv circular input: torch.Size([1, 256, 240, 180])
---- double conv circular pad: torch.Size([1, 256, 240, 182])
---- double conv circular after conv1: torch.Size([1, 64, 240, 180])
---- double conv circular after conv1 pad: torch.Size([1, 64, 240, 182])
---- double conv circular after conv2: torch.Size([1, 64, 240, 180])
++++ x conv: torch.Size([1, 64, 240, 180])
after up3  torch.Size([1, 64, 240, 180])
++++ up input: torch.Size([1, 64, 240, 180])
++++ up : torch.Size([1, 64, 480, 360])
++++ up diffY: 0
++++ up diffX: 0
++++ x1 pad: torch.Size([1, 64, 480, 360])
++++ x cat: torch.Size([1, 128, 480, 360])
---- double conv circular input: torch.Size([1, 128, 480, 360])
---- double conv circular pad: torch.Size([1, 128, 480, 362])
---- double conv circular after conv1: torch.Size([1, 64, 480, 360])
---- double conv circular after conv1 pad: torch.Size([1, 64, 480, 362])
---- double conv circular after conv2: torch.Size([1, 64, 480, 360])
++++ x conv: torch.Size([1, 64, 480, 360])
after up4  torch.Size([1, 64, 480, 360])
after last out conv  torch.Size([1, 512, 480, 360])
>> x:  torch.Size([1, 512, 480, 360])
>> permuted x:  torch.Size([1, 480, 360, 512])
>>reshaped x:  torch.Size([1, 480, 360, 32, 16])
>>final x:  torch.Size([1, 16, 480, 360, 32])
'''