import torch
import torch.nn as nn
import torch.optim as optim

class DownBlock(nn.Module):
    '''
    input=(B,in_chan,H,W) ----> output_pooled=(B,out_chan,H/2,W/2), output=(B,out_chan,H,W)
    '''
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.elu1 = nn.ELU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.elu2 = nn.ELU()
        
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.elu1(self.norm1(self.conv1(x)))
        #print(f"down conv1: {x.shape}")
        x = self.elu2(self.norm2(self.conv2(x)))
        #print(f"down conv2: {x.shape}")
        x_pooled = self.pool(x)
        #print(f"down pooled: {x_pooled.shape}")
        return x_pooled, x

class UpBlock(nn.Module):
    '''
    input=(B,in_chan,H,W), skip_input=(B, in_chan, H, W) ----> (B,out_chan,H*2,W*2)
    out_chans is supposed to be the same number of channels as the skip connection input

    which means the up block (decoder) outputs the same shape as the output of the corresponding encoder block. 
    '''
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv1 = nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.elu1 = nn.ELU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.elu2 = nn.ELU()
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = self.upsample(x)
        #print(f"up upsample trans conv: {x.shape}")
        x = torch.cat((x, skip), dim=1) #along the channel dimension
        #print(f"up skip conn: {x.shape}")
        x = self.elu1(self.norm1(self.conv1(x)))
        #print(f"up conv1: {x.shape}")
        x = self.elu2(self.norm2(self.conv2(x)))
        #print(f"up conv2: {x.shape}")
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(UNet, self).__init__()
        channels = [32, 64, 64, 128, 128]
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.down_blocks.append(DownBlock(prev_channels, ch))
            prev_channels = ch

        # Bottleneck
        bottleneck_dim = 128
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], bottleneck_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(bottleneck_dim),
            nn.ELU(),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(bottleneck_dim),
            nn.ELU()
        )

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        prev_channels = bottleneck_dim
        for ch in reversed(channels):
            self.up_blocks.append(UpBlock(prev_channels, ch))
            prev_channels = ch

        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.down_blocks):
            #print(f"down {i}")
            x, skip = down(x)
            skips.append(skip)
        
        x = self.bottleneck(x)
        
        for i, up in enumerate(self.up_blocks):
            #print(f"up {i}")
            x = up(x, skips[-(i + 1)])

        return self.final_conv(x)

# Model instantiation
model = UNet(in_channels=2, out_channels=2)

# Optimizer setup
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# For inference, the gradient update step, noise levels, and iterations could be set up separately

if __name__=="__main__":
    B,C,H,W = 1,2,1024, 1024
    x = torch.ones((B,C,H,W))
    # down = DownBlock(C, 256)
    # up = UpBlock(256, 256)
    
    # out_pooled, out = down(x) #(B,256,H/2,W/2), (B,256,H,W)
    # out= up(out_pooled, out)    


    model = UNet(in_channels=2, out_channels=2)
    out = model(x)
    