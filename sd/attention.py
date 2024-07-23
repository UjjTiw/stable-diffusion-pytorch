import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias= True, out_proj_bias= True):
        super().__init()
        
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed,d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
 
    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        #  (Batch, height*width, channels)
        input_shape = x.shape
        b, seq_len , d_model = input_shape
        internim_shape = (b, seq_len, self.n_heads, self.d_head)
        #(Batch, height*width, channels) -> (Batch, height*width, channels*3) -> (Batch, height*width, channels)
        q, k , v = self.in_proj(x).chunk(3, dim=-1)
         #  (Batch, height*width, channels) -> (Batch, height*width, H ,channels/H) -> (Batch, H, height* widht, channels/H)
        q = q.view(internim_shape).transpose(1, 2)
        k = k.view(internim_shape).transpose(1, 2)
        v = v.view(internim_shape).transpose(1, 2)
        # (Batch, H, height* widht, channels/H)->(Batch, H, height * width , height * width)
        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight = weight/math.sqrt(self.d_head)
        weight = torch.softmax(weight,dim = -1)
        # (Batch, H, height * width , height * width) @ (Batch, H, height* width, channels/H) -> (Batch, H, height* width, channels/H)
        output = weight @ k
        #  (Batch, H, height* width, channels/H) -> (Batch, height* width, H, channels/H) 
        output = output.transpose(1,2)
        # (Batch, height* width, H, channels/H) -> (Batch, height*width, channels)
        output = output.reshape(input_shape)

        output = self.out_proj(output)
        return output




        