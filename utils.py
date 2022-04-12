import numpy as np
import torch 
from model import Embedding, NeRF
import torch.optim as optim

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
    model_coarse = NeRF().to(args.device)
    model_fine = NeRF().to(args.device)
    optimizer = optim.Adam(list(model_coarse.parameters())+list(model_fine.parameters()), args.lr, betas=(0.9, 0.999))
    train_config = {
        "model_c" : model_coarse,
        "model_f" : model_fine,
        "batch_size": 1024 * 32,
    }
    return optimizer, train_config

def run(rays, model, batch_size):
    out_shape = list(rays.shape)
    rays = rays.reshape((-1, out_shape[-1]))
    print(rays.shape)
    out = model(rays)
    # out = torch.cat([model(rays[i:i+batch_size]) for i in range(0, rays.shape[0], batch_size)], dim = 0).reshape(out_shape[:-1]+[4])
    return out
    # return None

def output2rgb(output):
    
    pass

def hierarchical_sampling():
    pass
def render(rays_o, rays_d, near, far, N_samples, **train_config):

    N_rays = rays_o.shape[0]
    t = near + (far - near) * torch.linspace(0, 1, steps = N_samples)
    t = t.expand([N_rays, N_samples])
    rays = rays_o[:,None, :] + t[..., None] * rays_d[:,None,:]
    rays = torch.cat([rays, rays_d[:,None, :].expand(rays.shape)], dim = -1)
    print(rays.shape)
    output = run(rays, train_config['model_c'], train_config['batch_size'])

    return output, None