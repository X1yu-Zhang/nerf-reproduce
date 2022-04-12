import configargparse
from tqdm import tqdm, trange
from datasets import load_dataset, RaysDataset
from utils import render, create_model
from model import NeRF

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# def render():
    
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

    args.add_argument("--start", type=int, default=0)
    args.add_argument("-N_iters", type=int, default=20000)
    args.add_argument("--batch_size", type=int, default=128)

    args.add_argument("-lr", type=float, default=5e-4)
    args.add_argument("-lr_decay", type=float, default=250)

    args.add_argument("--test", type=int, default= 0)

    args = args.parse_args()

    return args

def main():
    args = config()
    print("Loading Data")
    ray_rgb_train, ray_rgb_test, ray_rgb_train, near, far = load_dataset(args)
    print("Load Done!")
    train_set = RaysDataset(ray_rgb_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    optimizer, train_config = create_model(args)
    start = args.start
    N_iters = args.N_iters
    global_step = start

    for idx in trange(start, N_iters + 1):
        for i, data in enumerate(train_loader):
            rays_o, rays_d, target = torch.split(data.float(), [3,3,3], dim = 1)            

            rgb, extras = render( rays_o, rays_d, near, far, args.N_samples, **train_config)
            break
            loss = (rgb-target)**2
            if 'coarse_rgb' in extras:
                loss += (extras['coarse_rgb'] - target) ** 2
                
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            decay_rate = 0.1
            decay_steps = args.lr_decay * 1000
            new_lr = args.lr * (decay_rate ** (global_step / decay_steps))

            global_step += 1
            if global_step > N_iters:
                break
        
if __name__ == "__main__":
    # torch.set_default_tensor_type("torch.cuda.FloatTensor")
    main()
