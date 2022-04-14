from re import I
import numpy as np
import torch 
from model import Embedding, NeRF
import torch.optim as optim
import configargparse

to8b = lambda x: (255 * np.clip(x, 0,1)).astype(np.uint8)
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
    args.add_argument("--device", type=str, default='cuda' )
    
    args.add_argument("--datadir", type=str, default='./data/nerf_synthetic/lego')
    args.add_argument("--dataset_type", default='blender', type=str)
    args.add_argument("--half_res", action="store_true")
    args.add_argument("--white_bkgd", action="store_true")
    args.add_argument("--ndc", action="store_true")
    
    args.add_argument("--N_samples", type=int, default=64)
    args.add_argument("--N_fine", type=int, default=128)
    args.add_argument("--noise", type=float, default=1)
    args.add_argument("--perturb", action="store_true")

    args.add_argument("--N_iters", type=int, default=20000)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--network_batch", type=int, default=1024)


    args.add_argument("--D", type=int, default=8)
    args.add_argument("--W", type=int, default=256)
    args.add_argument("--o_freq", type=int, default=10)
    args.add_argument("--d_freq", type=int, default=4)
    args.add_argument("-lr", type=float, default=5e-4)
    args.add_argument("log_sampling", action="store_true")

    args.add_argument("--lr_decay", type=float, default=250)
    args.add_argument("--i_test", type=int, default=100)
    args.add_argument("--i_val", type=int, default=200)

    args.add_argument("--test", type=int, default= 0)
    args.add_argument("--render_only", action="store_true")

    args.add_argument("--ckpt", type=str, default=None)
    args.add_argument("--output", type=str, default='./output')
    args = args.parse_args()
    
    render_config = get_render_config(args)
    model_config = get_model_config(args)
    dataset_config = get_dataset_config(args)
    train_config = get_train_config(args)

    return args, render_config, model_config, dataset_config, train_config

def get_rays_rgb(poses, images, K, near = None, ndc = False):
    H, W = images.shape[1:-1]
    xyz = get_xyz(H, W, K) # [W, H, 3]
    images = np.transpose(images, [0,2,1,3]) # [N, W, H, 3]
    xyz = np.broadcast_to(xyz, images.shape) # [N, W, H, 3]
    xyz = (poses[:, None, None, :3, :3] @ xyz[...,None]).squeeze(axis = -1) # [N, W, H, 3]
    d = xyz / np.linalg.norm(xyz, axis = -1)[...,None]
    # print(xyz.shape)
    origins = poses[..., -1]
    origins = origins[:, None, None, :]
    # print(origins.shape, xyz.shape)
    origins = np.broadcast_to(origins, xyz.shape)
    if ndc:
        assert near is not None
        focal = K[0,0]
        origins, xyz = ndc_transform(origins, xyz, focal, near, H, W)
    rays_rgb = np.concatenate([origins, xyz, d, images], axis = -1).reshape([-1,12])
    return rays_rgb

def get_rays_with_pose(H, W, K, pose, ndc = False, near = None):
        xyz = get_xyz(H, W, K)
        pose = pose[:3,:4]
        # print(pose.shape, xyz.shape)
        d = (pose[:3,:3] @ xyz[...,None]).squeeze(axis = -1)
        o = pose[None, None, :, 3]
        o = np.broadcast_to(o, d.shape)
        d_unit = d / np.linalg.norm(d, axis = -1, keepdims=True)
        if ndc:
            focal = K[0,0]
            o, d = ndc_transform(o, d, focal, near, H, W)

        rays = np.concatenate([o,d,d_unit], axis = -1).reshape([-1,9])
        return rays

def get_xyz(H, W, K):
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='ij')
    xy = np.stack([i,j], axis = -1)
    xyz = np.concatenate([xy, np.ones_like(xy[...,0, None])], axis = -1) - np.array([K[0,2], K[1,2], 0])# [W, H, 3]
    scale = np.array([K[0,0], -K[1,1], -1])
    xyz = xyz / scale
    return xyz


trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]],dtype=float)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], dtype=float)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], dtype=float)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def get_render_poses(phi = -30, radius = 4.0, n = 40):
    render_poses = np.stack([pose_spherical(angle, phi, radius) for angle in np.linspace(-180,180,n+1)[:-1]], 0)
    return render_poses

def get_render_config(args):
    config = {
        "N_samples":args.N_samples,
        "N_fine" : args.N_fine,
        "noise" : args.noise,
        "white_bkgd": args.white_bkgd,
        "batch_size" : args.network_batch,
        "perturb" : args.perturb,
        "ndc": args.ndc,
        "output"  : args.output,
        "test" : args.test
    }
    if args.render_only:
        config['noise'] = 0
    return config

def get_model_config(args):
    config = {
        "ckpt": args.ckpt,
        "D" : args.D,
        "W" : args.W,
        "o_freq": args.o_freq,
        "d_freq": args.d_freq,
        "log_sampling": args.log_sampling,
        "lr" : args.lr
    }
    return config

def get_train_config(args):
    config = {
        "N_iters" : args.N_iters,
        "lr": args.lr,
        "lr_decay" : args.lr_decay,
        "i_test" : args.i_test, 
        "i_val" : args.i_val
    }
    return config

def get_dataset_config(args):
    config = {
        "datadir": args.datadir,
        "dataset_type": args.dataset_type,
        "half_res" : args.half_res,
        "white_bkgd" : args.white_bkgd,
        "ndc": args.ndc,
        "test" : args.test,
        "render_only": args.render_only
    }
    return config

def create_model(train_config):
    ckpt = train_config['ckpt']
    D, W = train_config['D'], train_config['W']
    o_freq, d_freq = train_config['o_freq'], train_config['d_freq']
    log_sampling = train_config['log_sampling']
    lr = train_config['lr']
    model_coarse = NeRF(D, W, o_freq = o_freq, d_freq = d_freq, log_sampling = log_sampling)
    model_fine = NeRF(D, W, o_freq = o_freq, d_freq = d_freq, log_sampling = log_sampling)
    optimizer = optim.Adam(list(model_coarse.parameters())+list(model_fine.parameters()), lr, betas=(0.9, 0.999))
    global_step = None
    if ckpt != None:
        model_config = torch.load(ckpt)        
        global_step  = model_config['global_step']
        model_coarse.load_state_dict(model_config['network_fn_state_dict'])
        model_fine.load_state_dict(model_config['network_fine_state_dict'])
        optimizer.load_state_dict(model_config['optimizer_state_dict'])

    config = {
        "model_c" : model_coarse,
        "model_f" : model_fine,
    }
    return optimizer, config, global_step

def output2rgb(output, intervals, scale, noise_std = 0, white_bkgd = False):
    rgb, alpha = torch.split(output, [3,1], dim = -1)
    if noise_std > 0:
        alpha = alpha + torch.randn_like(alpha) * noise_std
    rgb = torch.sigmoid(rgb)
    delta = torch.cat([intervals[..., 1:] - intervals[...,:-1], torch.Tensor([1e10]).expand([intervals.shape[0],1])], dim = 1) 
    delta = delta * scale[...,None]
    alpha = torch.exp(-delta * torch.relu(alpha.squeeze()))
    T = torch.cat([torch.ones([alpha.shape[0],1]), torch.cumprod(alpha+1e-10, dim = 1)[...,:-1]], dim = -1)
    alpha = T * (1 - alpha)
    rgb = torch.sum(alpha[...,None] * rgb, dim = 1).squeeze()
    weight_sum = torch.sum(alpha, dim = -1)
    depth = torch.sum(alpha * intervals, dim = -1)
    if white_bkgd:
        rgb = rgb + (1 - weight_sum[...,None])
    return rgb, depth, alpha


def get_rays(rays_o, rays_d, d_unit, t):
    rays = rays_o[:, None, :] + t[..., None] * rays_d[:, None, :]
    return torch.cat([rays, d_unit[:,None, :].expand(rays.shape)], dim = -1)
