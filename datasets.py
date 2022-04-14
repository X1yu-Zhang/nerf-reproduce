import os
import cv2
import numpy as np
import imageio
import json
import torch
from torch.utils.data import Dataset
from utils import *
def load_blender_data(args):
    datadir = args['datadir']
    render_only = args['render_only']
    three_set = ['train', 'test', 'val'] if not render_only else ['train']
    datasets = {}
    H, W = None, None
    focal = None
    for keys in three_set:
        with open(os.path.join(datadir, 'transforms_{}.json'.format(keys))) as f:
            data = json.load(f) 

            images = []
            extrinctics = []
            for i, frame in enumerate(data['frames']):
                if args['test'] != 0 and i >= args['test']:
                    break
                file_path = frame['file_path']+'.png'
                matrix = np.array(frame['transform_matrix'], dtype=np.float32)
                image = np.array(imageio.imread(os.path.join(datadir, file_path))) / 255.
                if args['half_res']:
                    image = cv2.resize(image, (0,0), fx = 0.5, fy = 0.5)
                images.append(image)
                extrinctics.append(matrix)
                if render_only:
                    break
            images = np.stack(images, axis = 0)
            extrinctics = np.stack(extrinctics, axis = 0)[:,:3,:]

            if H is None and W is None:
                H, W = images.shape[1:-1]
            if args['white_bkgd']:
                images = images[..., :3] * images[..., -1:] + (1. - images[...,-1:])
            else:
                images = images[..., :3]
            datasets[keys] = {
                'images': images,
                'extrinsic_matrix': extrinctics
            }
            if focal is None:
                focal = 0.5 * W / np.tan(data['camera_angle_x'] / 2)

    near, far = 2., 6.
    K = np.array([
        [focal, 0, 0.5*W],
        [0,focal, 0.5*H],
        [0,0,1]
    ])
    render_poses = get_render_poses()
    return K, datasets, near, far, render_poses
    
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
    K, datasets, near, far, render_poses = eval('load_'+args['dataset_type']+'_data')(args)
    
    split = ['train', 'test', 'val']
    if args['render_only']:
        H, W = datasets['train']['images'].shape[1:-1]
        return H, W, K, near, far, render_poses
    else:
        for keys in split:
            poses = datasets[keys]['extrinsic_matrix']
            images = datasets[keys]['images']
            if keys == 'train':
                rays = get_rays_rgb(poses, images, K, near, ndc = args['ndc'])


        return rays, datasets['test'], datasets['val'] ,near, far