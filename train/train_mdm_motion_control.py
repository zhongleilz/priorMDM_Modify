# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""
import os
import json
from diffusion.respace import SpacedDiffusion
from utils.fixseed import fixseed
from utils.parser_util import train_inpainting_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import torch
from data_loaders.humanml_utils import get_inpainting_mask
from torch.utils.data import DataLoader
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
import numpy as np
def main():
    args = train_inpainting_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    length = 196
    if args.masked_rate in [1,2,5]:
        choose_seq_num = int(args.masked_rate)
    elif args.masked_rate in [25,100]:
        choose_seq_num = int(args.masked_rate*length*0.01)
    else:
        choose_seq_num = np.random.choice(length-1,1) + 1

    # print("choose_seq_num",choose_seq_num)
    choose_seq = np.random.choice(length,choose_seq_num,replace = False)
    choose_seq.sort()
    choose_mask = np.zeros((1, length))
    choose_mask[:,choose_seq] = 1
    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,
                              short_db=args.short_db, cropping_sampler=args.cropping_sampler)
    
    class InpaintingDataLoader(object):
        def __init__(self, data):
            self.data = data
        
        def __iter__(self):
            for motion, cond in super().__getattribute__('data').__iter__():
                cond['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, motion.shape,choose_mask,True)).to(motion.device)
                yield motion, cond
        
        def __getattribute__(self, name):
            return super().__getattribute__('data').__getattribute__(name)
        
        def __len__(self):
            return len(super().__getattribute__('data'))

    data = InpaintingDataLoader(data)

    print("creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion if args.filter_noise else SpacedDiffusion
    model, diffusion = create_model_and_diffusion(args, data, DiffusionClass=DiffusionClass)
    # This is the only required change to the original code
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    
    # FIXME explain
    class InpaintingTrainLoop(TrainLoop):
        def _load_optimizer_state(self):
            pass
    
    InpaintingTrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
