# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from diffusion.respace import SpacedDiffusion
from model.model_blending import ModelBlender
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import edit_inpainting_args
from utils.model_util import load_model_blending_and_diffusion
from utils import dist_util
from model.cfg_sampler import wrap_model
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml_utils import get_inpainting_mask
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion,plot_3d_motion_hint,plot_3d_motion_combine
import shutil

def main():
    args_list = edit_inpainting_args()
    args = args_list[0]
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    # n_frames = min(max_frames, int(args.motion_length*fps))
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.inpainting_mask, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    print('Loading dataset...')
    # args.batch_size = 6
    args.num_samples = 10

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = 8#args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              load_mode='train',
                              size=args.num_samples)  # in train mode, you get both text and motion.

    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions
    print("args.masked_rate",args.masked_rate)
    print("Creating model and diffusion...")

    length = 196
    if args.masked_rate in [1,2,5]:
        choose_seq_num = int(args.masked_rate)
    elif args.masked_rate in [25,90,100,99]:
        choose_seq_num = int(args.masked_rate*length*0.01)
    else:
        choose_seq_num = np.random.choice(length-1,1) + 1

    print("choose_seq_num",choose_seq_num)
    choose_seq = np.random.choice(length,choose_seq_num,replace = False)
    choose_seq.sort()
    choose_mask = np.zeros((1, length))
    choose_mask[:,choose_seq] = 1

    mask_idx = np.where(choose_mask == 0)
    print("mask_idx",mask_idx)
    DiffusionClass = InpaintingGaussianDiffusion if args_list[0].filter_noise else SpacedDiffusion
    model, diffusion = load_model_blending_and_diffusion(args_list, data, dist_util.dev(), DiffusionClass=DiffusionClass)

    iterator = iter(data)

    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions.to(dist_util.dev())
    if args.text_condition != '':
        texts = [args.text_condition] * args.num_samples
        model_kwargs['y']['text'] = texts

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion'] = input_motions

    
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape,choose_mask,True)).float().to(dist_util.dev())

    all_motions = []
    all_lengths = []
    all_text = []
    all_hint = []
    all_hint_for_vis = []

    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=input_motions,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        print("before sample",sample.shape)
        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            input_motions_ = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
            input_motions_ = recover_from_ric(input_motions_, n_joints)
            input_motions_ = input_motions_.view(-1, *input_motions_.shape[2:]).permute(0, 2, 3, 1)


        all_text += model_kwargs['y']['text']
        if 'hint' in model_kwargs['y']:
            hint = model_kwargs['y']['hint'].cuda()

            # denormalize hint
            from os.path import join as pjoin
            # data_root = '/work/vig/zhonglei/Control-mdm/dataset/KIT-ML'
            # raw_mean = torch.from_numpy(np.load(pjoin(data_root, 'Mean_raw.npy'))).cuda()
            # raw_std = torch.from_numpy(np.load(pjoin(data_root, 'Std_raw.npy'))).cuda()
            mask = hint.view(hint.shape[0], hint.shape[1], 21, 3).sum(-1) != 0

            # hint = hint * raw_std + raw_mean
            input_motions_ = input_motions_.permute(0,3,1,2).cuda() * mask.unsqueeze(-1)
            input_motions_ = input_motions_.view(input_motions_.shape[0], input_motions_.shape[1], -1)

            hint = hint.view(hint.shape[0], hint.shape[1], 21, 3) * mask.unsqueeze(-1)
            hint = hint.view(hint.shape[0], hint.shape[1], -1)
            
            diff = (input_motions_ - hint).mean()
                # ---
            
            hint = input_motions_
            all_hint.append(hint.data.cpu().numpy())
            hint = hint.view(hint.shape[0], hint.shape[1], 21, 3)
            input_motions_ = input_motions_.view(input_motions_.shape[0], input_motions_.shape[1], 21, 3)

            all_hint_for_vis.append(hint.data.cpu().numpy())

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if 'hint' in model_kwargs['y']:
        all_hint = np.concatenate(all_hint, axis=0)[:total_num_samples]
        all_hint_for_vis = np.concatenate(all_hint_for_vis, axis=0)[:total_num_samples]
    
    #all_motions (6, 21, 3, 196)

    from utils.simple_eval import simple_eval
    results = simple_eval(all_motions, all_hint)

    print("simple_eval",results)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    # np.save(npy_path,
    #         {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
    #          'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})

    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,"hint":all_hint_for_vis,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})

    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions, 21)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()


    for sample_i in range(args.batch_size):
        rep_files = []
        if args.show_input:
            caption = 'Input Motion'
            length = model_kwargs['y']['lengths'][sample_i]
            motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
            motion_gen = all_motions[sample_i].transpose(2, 0, 1)[:length]

            hint_vis_ = all_hint_for_vis[sample_i]

            mask = (hint_vis_[:,0,1] != 0).astype(int)

            diff = (motion - motion_gen)[:,0,:]

            percent = diff / motion_gen[:,0,:]

            # mask = motion_gen.sum(axis=0, keepdims=True) != 0

            loss = np.linalg.norm((motion[:,0,:] - motion_gen[:,0,:]), axis=0)
            loss = loss.sum() / mask.sum()

            save_file = 'input_motion{:02d}.mp4'.format(sample_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)

            print(f'[({sample_i}) "{caption}" | -> {save_file}]')
            plot_3d_motion_combine(animation_save_path, skeleton, motion,motion_gen, title=caption,
                        dataset=args.dataset, fps=fps, vis_mode='gt',
                        gt_frames=gt_frames_per_sample.get(sample_i, []))

        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            if args.guidance_param == 0:
                caption = 'Edit [{}] unconditioned'.format(args.inpainting_mask)
            else:
                caption = 'Edit [{}]: {}'.format(args.inpainting_mask, caption)
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]

            if 'hint' in model_kwargs['y']:
                hint = all_hint_for_vis[rep_i*args.batch_size + sample_i]
            else:
                hint = None

            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
         
            # plot_3d_motion_hint(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps, hint=hint)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        if args.num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions + (1 if args.show_input else 0)} '
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)
            print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()
