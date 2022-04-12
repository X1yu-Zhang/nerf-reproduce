from tqdm import tqdm, trange
from datasets import load_dataset, RaysDataset
from utils import render, create_model, config

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# def render():
    


def main(args):
    print("Loading Data")
    ray_rgb_train, near, far = load_dataset(args)
    print("Load Done!")
    train_set = RaysDataset(ray_rgb_train)
    train_loader = iter(DataLoader(train_set, batch_size=args.batch_size, shuffle=True))
    optimizer, train_config = create_model(args)
    start = args.start
    N_iters = args.N_iters
    global_step = start
    for idx in trange(start, N_iters):
        data = next(train_loader)
        rays_o, rays_d, target = torch.split(data.float(), [3,3,3], dim = 1)            
        rgb, depth, rgb_c = render( rays_o, rays_d, near, far, args.N_samples, **train_config)
        loss = torch.mean((rgb-target)**2)
        loss += torch.mean((rgb_c - target) ** 2) 
            
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lr_decay * 1000
        new_lr = args.lr * (decay_rate ** (global_step / decay_steps))

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        global_step += 1
        if global_step > N_iters:
            break

def render_only(args):
    ckpt = args.ckpt
    if ckpt == None:
        print("Input the path of pretrained model")
        return

    
    pass
if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    args = config()
    if args.render_only:
        render_only(args)
    else:
        main(args)
