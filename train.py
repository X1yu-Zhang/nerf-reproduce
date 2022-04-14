import imageio
import os
from tqdm import tqdm, trange
from datasets import load_dataset, RaysDataset
from utils import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def run(rays, model, batch_size):
    out_shape = list(rays.shape)
    rays = rays.reshape((-1, out_shape[-1]))
    # out = model(rays).reshape(out_shape[:-1]+[4])
    out = torch.cat([model(rays[i:i+batch_size]) for i in range(0, rays.shape[0], batch_size)], dim = 0).reshape(out_shape[:-1]+[4])
    out = out.reshape(out_shape[:-1]+[4])
    return out   

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

def render(rays_o, rays_d, d_unit, near, far, **config):
    noise = config["noise"]
    N_samples = config['N_samples']
    white_bkgd = config["white_bkgd"]

    N_rays = rays_o.shape[0]

    if config['perturb']:
        t = torch.linspace(0,1, N_samples + 1)
        t = torch.rand(N_samples) / N_samples + t[:N_samples]
    else:
        t = torch.linspace(0, 1, steps = N_samples)
    t = near + (far - near) * t
    t = t.expand([N_rays, N_samples])
    # coarse 
    rays = get_rays(rays_o, rays_d, d_unit, t)

    output = run(rays, config['model_c'], config['batch_size'])
    rgb_c, depth_c, alpha = output2rgb(output, t, torch.norm(rays_d, dim = -1),noise, white_bkgd)

    # Hierarchical Sampling

    # samples = hierarchical_sampling(alpha[...,:-1], t, config['N_fine'])
    samples = hierarchical_sampling(alpha[...,1:-1],(t[...,1:]+t[...,:-1]) / 2, config['N_fine'])
    samples = samples.detach()
    # fine
    t , _= torch.sort(torch.cat([t, samples], dim = 1), dim = 1)

    rays_fine = get_rays(rays_o, rays_d, d_unit, t)

    output = run(rays_fine, config['model_f'], config['batch_size'])
    rgb, depth, alpha = output2rgb(output, t, torch.norm(rays_d, dim = -1), noise, white_bkgd)

    return rgb, depth, rgb_c

def render_only(render_config, model_config, dataset_config):
    ckpt = args.ckpt
    output = args.output
    if ckpt == None:
        print("ERROR: Input the path of pretrained model")
        return

    print(dataset_config)
    H, W, K, near, far, render_poses = load_dataset(dataset_config)
    _, models, _ = create_model(model_config)
    render_config.update(models)
    batch_size = args.batch_size
    
    for ith, pose in enumerate(tqdm(render_poses)):
        rgbs = []
        rays = get_rays_with_pose(H, W, K, pose, render_config['ndc'], near)
        for i in range(0, rays.shape[0], batch_size):
            rays_o, rays_d, d_unit = torch.split(torch.Tensor(rays[i:i+batch_size]), [3,3,3], dim = 1)
            rgb, _, _ = render(rays_o, rays_d, d_unit, near, far, **render_config)
            rgb = rgb.cpu().numpy()
            rgbs.append(rgb)
        rgbs = np.concatenate(rgbs, axis = 0) 
        rgbs = rgbs.reshape([W,H,3]).transpose([1,0,2])

        if output is not None:
            if not os.path.exists(output):
                os.makedirs(output)
            rgbs = to8b(rgbs)       
            filename = os.path.join(output, '{:03d}.png'.format(ith))
            imageio.imwrite(filename, rgbs)
    # print(H, W, K, near, far)
    
def main(render_config, model_config, dataset_config, train_config):
    print("Loading Data")
    ray_rgb_train, test_set, val_set, near, far = load_dataset(dataset_config)
    print("Load Done!")
    train_set = RaysDataset(ray_rgb_train)
    train_loader = iter(DataLoader(train_set, batch_size=args.batch_size, shuffle=True))
    optimizer, config, start = create_model(model_config)
    render_config.update(config)
    if start is None:
        start = 0

    N_iters = train_config['N_iters']
    lr = train_config['lr']
    lr_decay = train_config['lr_decay']
    i_test = train_config['i_test']
    i_val = train_config['i_val']

    global_step = start
    for idx in trange(start, N_iters):
        data = next(train_loader)
        rays_o, rays_d, d_unit, target = torch.split(data.float(), [3,3,3,3], dim = 1)            
        rgb, depth, rgb_c = render( rays_o, rays_d, d_unit, near, far, **render_config)
        loss = torch.mean((rgb-target)**2)
        loss += torch.mean((rgb_c - target) ** 2) 
            
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        decay_rate = 0.1
        decay_steps = lr_decay * 1000
        new_lr = lr * (decay_rate ** (global_step / decay_steps))

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        # if idx % i_test:
            
        # if idx % i_val: 
            
        global_step += 1
        if global_step > N_iters:
            break


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    args, render_config, model_config, dataset_config, train_config = config()
    if args.render_only:
        with torch.no_grad():
            render_only(render_config, model_config, dataset_config)
    else:
        main(render_config, model_config, dataset_config, train_config)
