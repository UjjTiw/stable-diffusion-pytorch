import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.group_norm_1 = nn.GroupNorm(32,in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3)

        self.group_norm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, padding=1, kernel_size=3)

        if in_channels==out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, padding=0, kernel_size=1)
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        # x: (Batch, In, H , W)
        residue = x
        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.group_norm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(x)
    
class VAE_Attention(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention()

    def forward(self, x:torch.Tensor):
        # x : (Batch, channels, height, width)
        residue = x
        # x : (Batch, channels, height, width)
        x = self.groupnorm(x)

        n,c,h,w = x.shape 
        # x : (Batch, channels, height, width) -->(Batch, channels, height * width)
        x = x.view((n,c,h*w))
        # (Batch, channels, height * width) -> (Batch, height*width, channels)
        x = x.transpose(-1,-2)
        #  (Batch, height*width, channels)
        x  = self.attention(x)
        # (Batch, height*width, channels)-->(Batch, channels, height * width) -> 
        x = x.transpose(-1,-2)
        # x : (Batch, channels, height * width)-->(Batch, channels, height, width) 
        x = x.view((n,c,h,w))

        return x + residue

class VAE_Decoder(nn.Sequential):
    def __init_subclass__(self):
        return super().__init_subclass__(
            nn.Conv2d(4,4, kernel_size=1, padding=1),
            nn.Conv2d(4, 512, padding=1, kernel_size=3, padding_mode=1),
            VAE_ResidualBlock(512, 512),
            VAE_Attention(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, padding=1, kernel_size=3),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, padding=1, kernel_size=3),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128,3, padding=1,kernel_size=3)
        )
    
    def forward(self, x:torch.Tensor):
        x /= 0.18125
        for module in self:
            x = module(x)
        return x
       