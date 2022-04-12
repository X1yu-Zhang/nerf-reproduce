from re import I
import numpy as np
import torch 
from model import Embedding, NeRF
import torch.optim as optim
import configargparse

def ndc_transform(o, d, focal, near, H, W):
    t = -(near + o[..., 2]) / d[..., 2]
    o = o + t[..., None] * d
    o0 = -focal*2/W*(o[..., 0] / o[..., 2])
    o1 = -focal*2/H*(o[..., 1] / o[..., 2])
    o2 = 1 + 2 * near / o[...,2]

    d0 = -focal*2/W*(d[..., 0] / d[..., 2]) - o0
    d1 = -focal*2/H*(d[..., 1] / d[..., 2]) - o1
    d2 =  1 - o2

    o = np.stack([o0,o1,o2], axis=-1)
    d = np.stack([d0,d1,d2], axis=-1)
    return o, d

def config():
    args = configargparse.ArgumentParser()
    args.add_argument("--config", is_config_file=True) 
    args.add_argument("--name", type=str)
    args.add_argument("--basedir", type=str, default='./ckpt/')
    args.add_argument("--device", type=str, default='cuda' )
    
    args.add_argument("--datadir", type=str, default='./data/nerf_synthetic/lego')
    args.add_argument("--dataset_type", default='blender', type=str)
    args.add_argument("--half_res", action="store_true")
    args.add_argument("--white_bkgd", action="store_true")
    args.add_argument("--no_ndc", action="store_true")
    
    args.add_argument("--N_samples", type=int, default=64)
    args.add_argument("--N_fine", type=int, default=128)
    args.add_argument("--noise", type=float, default=1)

    args.add_argument("--start", type=int, default=0)
    args.add_argument("-N_iters", type=int, default=20000)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--network_batch", type=int, default=1024)

    args.add_argument("-lr", type=float, default=5e-4)
    args.add_argument("-lr_decay", type=float, default=250)

    args.add_argument("--test", type=int, default= 0)
    args.add_argument("--render_only", action="store_true")

    args.add_argument("--ckpt", type=str, default=None)
    args = args.parse_args()

    return args

def get_rays_rgb(poses, images, K, near = None, no_ndc = False):
    H, W = images.shape[1:-1]
    xy = np.mgrid[:W, :H]
    xyz = np.concatenate([xy, np.ones_like(xy[None, 0,..., ])], axis = 0) # [3, W, H]
    xyz = np.transpose(xyz, [1,2,0]) # [W, H, 3] xyz
    images = np.transpose(images, [0,2,1,3]) # [N, W, H, 3]
    xyz = np.broadcast_to(xyz, images.shape) # [N, W, H, 3]
    xyz = K @ xyz[..., None] # [N, W, H, 3, 1]
    xyz = np.concatenate([xyz, np.ones(list(xyz.shape[:-2])+[1,1])], axis = -2) # [N, W, H, 4, 1]
    xyz = (poses[:, None, None, ...] @ xyz).squeeze(axis = -1) # [N, W, H, 3, 1]
    xyz = xyz / np.linalg.norm(xyz, axis = -1)[...,None]
    # print(xyz.shape)
    origins = poses[..., -1]
    origins = origins[:, None, None, :]
    # print(origins.shape, xyz.shape)
    origins = np.broadcast_to(origins, xyz.shape)
    if not no_ndc:
        assert near is not None
        focal = K[0,0]
        origins, xyz = ndc_transform(origins, xyz, focal, near, H, W)
    rays_rgb = np.concatenate([origins, xyz, images], axis = -1).reshape([-1,9])
    return rays_rgb

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype=float)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]],dtype=float)

rot_theta = lambda th : np.array([
    [np.cos(th),-np.sin(th),0,0],
    [np.sin(th),np.cos(th),0,0],
    [0,0,1,0],
    [0,0,0,1]],dtype=float)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    return c2w

def get_render_poses(phi = -30, radius = 4.0, n = 40):
    render_poses = np.stack([pose_spherical(angle, phi, radius) for angle in np.linspace(-180,180,n+1)[:-1]], 0)
    return render_poses

def create_model(args):
    model_coarse = NeRF()
    model_fine = NeRF()
    optimizer = optim.Adam(list(model_coarse.parameters())+list(model_fine.parameters()), args.lr, betas=(0.9, 0.999))
    train_config = {
        "model_c" : model_coarse,
        "model_f" : model_fine,
        "N_fine" : args.N_fine,
        "noise" : args.noise,
        "white_bkgd": args.white_bkgd,
        "batch_size" : args.network_batch
    }
    return optimizer, train_config

def run(rays, model, batch_size):
    out_shape = list(rays.shape)
    rays = rays.reshape((-1, out_shape[-1]))
    # out = model(rays).reshape(out_shape[:-1]+[4])
    out = torch.cat([model(rays[i:i+batch_size]) for i in range(0, rays.shape[0], batch_size)], dim = 0).reshape(out_shape[:-1]+[4])
    out = out.reshape(out_shape[:-1]+[4])
    return out
    # return None

def output2rgb(output, intervals, scale, noise_std = 0, white_bkgd = False):
    rgb, alpha = torch.split(output, [3,1], dim = -1) # [N_rays, N_samples, 4]

    if noise_std > 0:
        alpha = alpha + torch.randn_like(alpha) * noise_std
    rgb = torch.sigmoid(rgb)
    delta = torch.cat([intervals[..., 1:] - intervals[...,:-1], torch.Tensor([1e10]).expand([intervals.shape[0],1])], dim = 1) * scale[..., None]
    alpha = torch.exp(-delta * torch.relu(alpha.squeeze()))
    T = torch.cat([torch.ones([alpha.shape[0],1]), torch.cumprod(alpha, dim = 1)[...,:-1]], dim = -1)
    alpha = T * (1 - alpha)

    rgb = torch.sum(alpha[...,None] * rgb, dim = 1).squeeze()
    weight_sum = torch.sum(alpha, dim = -1)
    depth = torch.sum(alpha * intervals, dim = -1)
    if white_bkgd:
        rgb = rgb + (1 - weight_sum)

    return rgb, depth, alpha

def hierarchical_sampling(weight, bins, N_samples):
    pdf = weight / (1e-5 + torch.sum(weight, dim = -1).unsqueeze(1) )
    cdf = torch.cat([torch.zeros([weight.shape[0],1]), torch.cumsum(pdf, dim = -1)], dim = -1)
    sample_y = torch.rand(list(cdf.shape[:-1])+[N_samples])
    idx_x = torch.searchsorted(cdf, sample_y, right = True)
    left = torch.max(torch.zeros_like(idx_x), idx_x-1)
    right = torch.min(torch.ones_like(idx_x) * (cdf.shape[-1] - 1), idx_x) 
    idx = torch.stack([left, right], dim = -1) # [N_rays, N_samples, 2]

    expand_shape = list(idx.shape[:-1]) + [cdf.shape[-1]]
    sample_interval = torch.gather(bins[:,None,:].expand(expand_shape), 2, idx)
    cdf_interval = torch.gather(cdf[:,None, :].expand(expand_shape), 2, idx)

    proportion_interval = cdf_interval[...,1] - cdf_interval[...,0]
    proportion_interval[proportion_interval < 1e-5] = 1
    proportion = (sample_y - cdf_interval[...,0]) / proportion_interval

    samples = sample_interval[..., 0] + proportion * ( sample_interval[...,1] - sample_interval[..., 0] )
    return samples

def get_rays(rays_o, rays_d, t):
    rays = rays_o[:, None, :] + t[..., None] * rays_d[:, None, :]
    return torch.cat([rays, rays_d[:,None, :].expand(rays.shape)], dim = -1)

def render(rays_o, rays_d, near, far, N_samples, **config):
    noise = config["noise"]
    white_bkgd = config["white_bkgd"]

    N_rays = rays_o.shape[0]

    t = torch.linspace(0, 1, steps = N_samples+1)
    t = torch.rand(N_samples) / N_samples + t[:N_samples]
    t = near + (far - near) * t
    t = t.expand([N_rays, N_samples])
    # coarse 
    rays = get_rays(rays_o, rays_d, t)
    output = run(rays, config['model_c'], config['batch_size'])
    rgb_c, depth_c, alpha = output2rgb(output ,t, torch.norm(rays_d, dim = -1),noise, white_bkgd)
    # Hierarchical Sampling
    samples = hierarchical_sampling(alpha[...,1:-1],(t[...,1:]+t[...,:-1]) / 2, config['N_fine'])
    samples = samples.detach()
    # fine
    t , _= torch.sort(torch.cat([t, samples], dim = 1), dim = 1)

    rays_fine = get_rays(rays_o, rays_d, t)

    output = run(rays_fine, config['model_f'], config['batch_size'])
    rgb, depth, alpha = output2rgb(output, t, torch.norm(rays_d, dim = -1), noise, white_bkgd)

    return rgb, depth, rgb_c