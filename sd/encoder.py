import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_Attention, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().init(
            # (Batch, 3, heigh, weight) --> (BAtch, 128, height, W)
            nn.Conv2d(3,128, kernel_size=3, padding=1),
            # (Batch, 128, heigh, weight) --> (BAtch, 128, height, W)
            VAE_ResidualBlock(128, 128),
            # (Batch, 128, heigh, weight) --> (BAtch, 128, height, W)
            VAE_ResidualBlock(128,128),
            # (Batch, 128, heigh, weight) --> (BAtch, 128, height/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=2),
            # (Batch, 128, height/2, weight/2) --> (BAtch, 256, height/2, W/2)
            VAE_ResidualBlock(128, 256),
            # (Batch, 256, heigh, weight) --> (BAtch, 256, heigh/2, W/2)
            VAE_ResidualBlock(256, 256),
            #(BAtch, 256, heigh/2, W/2)--> (BAtch, 256, heigh/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=2),
            #(BAtch, 256, heigh/4, W/4) --> (BAtch, 512, heigh/4, W/4)
            VAE_ResidualBlock(256, 512),
            # (BAtch, 512, heigh/4, W/4) --> (BAtch, 512, heigh/4, W/4)
            VAE_ResidualBlock(512, 512),
            # (BAtch, 512, heigh/4, W/4) --> (BAtch, 512, heigh/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, padding=0, stride=2),
            # (BAtch, 512, heigh/8, W/8) --> (BAtch, 512, heigh/8, W/8)
            VAE_ResidualBlock(512, 512),
            # (BAtch, 512, heigh/8, W/8) --> (BAtch, 512, heigh/8, W/8)
            VAE_ResidualBlock(512, 512),
            # (BAtch, 512, heigh/8, W/8) --> (BAtch, 512, heigh/8, W/8)
            VAE_ResidualBlock(512, 512),
            # (BAtch, 512, heigh/8, W/8) --> (BAtch, 512, heigh/8, W/8)
            VAE_Attention(512),
            # (BAtch, 512, heigh/8, W/8) --> (BAtch, 512, heigh/8, W/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512), 

            nn.SiLU(),
            # (BAtch, 512, heigh/8, W/8) --> (BAtch, 8, heigh/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
             # (BAtch, 8, heigh/8, W/8) --> (BAtch, 8, heigh/8, W/8)
            nn.Conv2d(8,8, kernel_size=1, padding=0)

        )
    
    def forward(self, x:torch.Tensor, noise:torch.Tensor):
        for module in self:
            if getattr(module, 'padding', None) == 2:
                x = F.pad(x, pad=(0,1,0,1))
            x = module(x)
        # (batch, 8, height/8, w/8) --> (batch, 4, height/8, w/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log_variance
        log_variance = torch.clamp(log_variance, -10, 30)
        # Compute the vairance 
        variance = log_variance.exp()
        # Compute the std deviation 
        stdev = variance.sqrt()
        # Transform the noise gaussian distribution 
        x = mean + stdev * noise
        # Scale the output(historical cause)
        x *= 0.18215

        return x
    