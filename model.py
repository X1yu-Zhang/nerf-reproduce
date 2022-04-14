import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding:
    def __init__(self, input_dim, length, include_input = True, log_sampling = True):
        super(Embedding, self).__init__()
        self.input_dim = input_dim
        self.length = length
        self.functs = [torch.sin, torch.cos]
        self.output_dim = 2 * length * input_dim + include_input * self.input_dim
        if log_sampling:
            self.freq = torch.pow(2, torch.linspace(0, length - 1, steps = length) )
        else:
            self.freq = torch.linspace(1, 2**(length-1), steps = length)

    def embed(self, x):
        ##  x: [N_rays, 3]
        ## self.freq: [length]
        
        embed_vec = x[..., None] * self.freq # [N_rays, 3, length]
        embed_vec = torch.stack([func(embed_vec) for func in self.functs], dim = -1) # [N_rays, 3, length, 2]
        embed_vec = embed_vec.permute([0,2,3,1]).reshape([embed_vec.shape[0], -1])  # [N_rays, length, 2, 3] [N_rays, 3 * 2 * length]
        x = torch.cat([x, embed_vec], dim = -1)
        return x

class NeRF(nn.Module):
    def __init__(self, D = 8, W = 256, input_ch = 3, input_ch_view = 3, output_ch = 4, skip_connect = [4], o_freq = 10, d_freq = 4, log_sampling = True, device='cuda'):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.device = device
        self.o_embedding = Embedding(input_ch, o_freq, log_sampling=log_sampling)
        self.d_embedding = Embedding(input_ch, d_freq, log_sampling=log_sampling)
        self.input_ch = self.o_embedding.output_dim
        self.input_ch_view = self.d_embedding.output_dim
        self.skip = skip_connect
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] + \
            [nn.Linear(self.W, self.W) if i not in skip_connect else nn.Linear(self.W+self.input_ch, W) for i in range(D-1)]
        )
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        # self.rgb_linears = nn.ModuleList(
        #     [nn.Linear(W+self.input_ch_view, W//2), nn.Linear(W//2, 3)]
        # )
        self.rgb_linear = nn.Linear(W//2,3)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view+W, W//2)])
        
    def forward(self, rays):  
        o, d = torch.split(rays, [3,3], dim = 1)
        o = self.o_embedding.embed(o)
        d = self.d_embedding.embed(d)
        h = o
        for i, model in enumerate(self.pts_linears):
            h = F.relu(model(h))
            if i in self.skip:
                h = torch.cat([o, h], -1)        
        alpha = self.alpha_linear(h)

        feature = self.feature_linear(h)
        h = torch.cat([feature, d], -1)
    
        for i, model in enumerate(self.views_linears):
            h = F.relu(model(h))

        rgb = self.rgb_linear(h)
        return torch.cat([rgb, alpha], dim = -1)
      
