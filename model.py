from unittest import skip
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, input_dim, length):
        super(Embedding, self).__init__()
        self.input_dim = input_dim
        self.length = length
        self.functs = [torch.sin, torch.cos]
        self.output_dim = 2 * length * input_dim
        self.freq = torch.pow(2, torch.linspace(0, length - 1, steps = length) )
    def forward(self, x):
        ##  x: [N_rays, 3]
        ## self.freq: [length]
        ## [N_rays, 3, length
        x = x[..., None] * self.freq # [N_rays, 3, length]
        x = torch.stack([func(x) for func in self.functs], dim = -2) # [N_rays, 3, 2, length]
        x = x.permute([0,1,3,2]).reshape([x.shape[0], -1])  # [N_rays, 3 * 2 * length]
        return x

class NeRF(nn.Module):
    def __init__(self, D = 8, W = 256, input_ch = 3, input_ch_view = 3, output_ch = 4, skip_connect = [4], xyz_freq = 4, direction_freq = 10):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.xyz_embedding = Embedding(input_ch, xyz_freq)
        self.dir_embedding = Embedding(input_ch, direction_freq)
        self.input_ch = input_ch
        self.input_ch_view = input_ch_view
        self.xyz_embed_dim = self.xyz_embedding.output_dim
        self.dir_embed_dim = self.dir_embedding.output_dim
        self.skip = skip_connect
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.xyz_embed_dim, self.W)] + \
            [nn.Linear(self.W, self.W) if i not in skip_connect else nn.Linear(self.W+self.xyz_embed_dim, W) for i in range(D-1)]
        )
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_forword = nn.Linear(W, W)
        self.rgb_linears = nn.ModuleList(
            [nn.Linear(W+self.dir_embed_dim, W//2), nn.Linear(W//2, 3)]
        )
        
    def forward(self, x):
        xyz, dir = torch.split(x, [self.input_ch, self.input_ch_view], dim=1 )
        
        with torch.no_grad():
            embed_xyz = self.xyz_embedding(xyz)
            embed_dir = self.dir_embedding(dir) 

        tmp = embed_xyz
        for i, model in enumerate(self.pts_linears):
            tmp = F.relu(model(tmp))
            if i in self.skip:
                tmp = torch.cat([embed_xyz, tmp], -1)        
        alpha = self.alpha_linear(tmp)
        xyz_feature = self.feature_forword(tmp)
        feature = torch.cat([xyz_feature, embed_dir],dim = -1 )
        tmp = feature
        for i, model in enumerate(self.rgb_linears):
            tmp = F.relu(model(tmp))
        
        return torch.cat([tmp,alpha], dim = -1)
        
