from ctypes.wintypes import tagRECT
from unittest import skip
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding:
    def __init__(self, input_dim, length):
        super(Embedding, self).__init__()
        self.input_dim = input_dim
        self.length = length
        self.functs = [torch.sin, torch.cos]
        self.output_dim = 2 * length * input_dim
        self.freq = torch.pow(2, torch.linspace(0, length - 1, steps = length) )
    def embed(self, x):
        ##  x: [N_rays, 3]
        ## self.freq: [length]
        ## [N_rays, 3, length
        x = x[..., None] * self.freq # [N_rays, 3, length]
        x = torch.stack([func(x) for func in self.functs], dim = -1) # [N_rays, 3, 2, length]
        x = x.permute([0,1,3,2]).reshape([x.shape[0], -1])  # [N_rays, 3, length, 2] [N_rays, 3 * 2 * length]
        return x

class NeRF(nn.Module):
    def __init__(self, D = 8, W = 256, input_ch = 3, input_ch_view = 3, output_ch = 4, skip_connect = [4], xyz_freq = 4, direction_freq = 10, batch_size = 64, device='cuda'):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.device = device
        self.xyz_embedding = Embedding(input_ch, xyz_freq)
        self.dir_embedding = Embedding(input_ch, direction_freq)
        self.input_ch = self.xyz_embedding.output_dim
        self.input_ch_view = self.dir_embedding.output_dim

        self.skip = skip_connect
        self.batch_size = batch_size
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] + \
            [nn.Linear(self.W, self.W) if i not in skip_connect else nn.Linear(self.W+self.input_ch, W) for i in range(D-1)]
        )
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_forword = nn.Linear(W, W)
        self.rgb_linears = nn.ModuleList(
            [nn.Linear(W+self.input_ch_view, W//2), nn.Linear(W//2, 3)]
        )
        
    def forward(self, rays):  
        xyz, dir = torch.split(rays, [3,3], dim = 1)
        xyz = self.xyz_embedding.embed(xyz)
        dir = self.dir_embedding.embed(dir)
        hidden = xyz
        for i, model in enumerate(self.pts_linears):
            hidden = F.relu(model(hidden))
            if i in self.skip:
                hidden = torch.cat([xyz, hidden], -1)        
        alpha = self.alpha_linear(hidden)
        feature = self.feature_forword(hidden)
        hidden = torch.cat([feature, dir],dim = -1 )
        del feature
        for i, model in enumerate(self.rgb_linears):
            if i == 0:
                hidden = model(hidden)
            else:
                hidden = model(F.relu(hidden))
        return torch.cat([hidden,alpha], dim = -1)
      
        
