import os
import cv2
import numpy as np
import imageio
import json
import torch
from torch.utils.data import Dataset
from utils import *
def load_blender_data(args):
    datadir = args.datadir
    three_set = ['train', 'test', 'val']
    datasets = {}
    H, W = None, None
    focal = None
    for keys in three_set:
        with open(os.path.join(datadir, 'transforms_{}.json'.format(keys))) as f:
            data = json.load(f) 
            if focal is None:
                focal = data['camera_angle_x']
            images = []
            extrinctics = []
            for i, frame in enumerate(data['frames']):
                if args.test != 0 and i >= args.test:
                    break
                file_path = frame['file_path']+'.png'
                matrix = np.array(frame['transform_matrix'], dtype=np.float32)
                image = np.array(imageio.imread(os.path.join(datadir, file_path))) / 255.
                if args.half_res:
                    image = cv2.resize(image, (0,0), fx = 0.5, fy = 0.5)
                images.append(image)
                extrinctics.append(matrix)
            images = np.stack(images, axis = 0)
            extrinctics = np.stack(extrinctics, axis = 0)[:,:3,:]

            if H is None and W is None:
                H, W = images.shape[1:-1]
            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[...,-1:])
            else:
                images = images[..., :3]
            datasets[keys] = {
                'images': images,
                'extrinsic_matrix': extrinctics
            }
    if args.half_res:
        focal = focal / 2.        
        H = H // 2
        W = W // 2
    near, far = 2., 6.
    K = np.array([
        [focal, 0, 0.5*W],
        [0,focal, 0.5*H],
        [0,0,1]
    ])
    return K, datasets, near, far
    
def load_deepvoxels_data(args):
    pass

def load_LINEMOD_data(args):
    
    pass

def load_llff_data(args):
    pass

class RaysDataset(Dataset):
    def __init__(self, rays):
        self.rays = rays
        self.len = rays.shape[0]
        pass    

    def __getitem__(self, index):
        return self.rays[index]

    def __len__(self):
        return self.len

def load_dataset(args):
    K, datasets, near, far = eval('load_'+args.dataset_type+'_data')(args)
    split = ['train', 'test', 'val']
    for keys in split:
        poses = datasets[keys]['extrinsic_matrix']
        images = datasets[keys]['images']
        if keys == 'train':
            rays = get_rays_rgb(poses, images, K, near, no_ndc = args.no_ndc)


    return rays, near, far