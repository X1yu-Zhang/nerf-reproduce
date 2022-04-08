import configargparse
from datasets import load_dataset
def config():
    args = configargparse.ArgumentParser()
    args.add_argument("--config", is_config_file=True) 
    args.add_argument("--name", type=str)
    args.add_argument("--basedir", type=str, default='./ckpt/')
    args.add_argument("--datadir", type=str, default='./data/blender/lego')

    args.add_argument("--N_samples", type=int, default=64)
    args.add_argument("--N_fine", type=int, default=0)
    
    args = args.parse_args()

    return args

def main():
    args = config()
    images, poses, near, far, render_poses, [H, W, focal], i_train, i_test, i_val = load_dataset(args)
    
    
    
if __name__ == "__main__":
    main()
