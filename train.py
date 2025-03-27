import argparse
import json
import os
import re
import sys
import torch
from torch_utils import distributed as dist
import dnnlib
from training_amed import training_loop
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


def parse_args():
    parser = argparse.ArgumentParser(description='Train AMED model')
    parser.add_argument('--dataset_name', default='celebahq', help='Name of the dataset')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--total_kimg', type=int, default=10, help='Total training kilo-images')
    parser.add_argument('--sampler_stu', default='amed', help='Student sampler')
    parser.add_argument('--sampler_tea', default='heun', help='Teacher sampler')
    parser.add_argument('--num_steps', type=int, default=4, help='Number of steps')
    parser.add_argument('--M', type=int, default=1, help='M parameter')
    parser.add_argument('--afs', type=bool, default=True, help='Use analytical first step')
    parser.add_argument('--scale_dir', type=float, default=0.01, help='Scale direction')
    parser.add_argument('--scale_time', type=float, default=0.0, help='Scale time')
    parser.add_argument('--schedule_type', default='time_uniform', help='Schedule type')
    parser.add_argument('--schedule_rho', type=float, default=1.0, help='Schedule rho')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--checkpoint_path', default='/kaggle/working/p2w/checkpoints/64x64.pt', help='Path to checkpoint')
    parser.add_argument('--guidance_type', default=None, help='Guidance type')
    parser.add_argument('--guidance_rate', type=float, default=0.0, help='Guidance rate')
    parser.add_argument('--prompt_path', default=None, help='Prompt path')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--batch_gpu', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--bench', type=bool, default=True, help='Enable cudnn benchmark')
    parser.add_argument('--max_order', type=int, default=3, help='Max order for solver')
    parser.add_argument('--predict_x0', type=bool, default=True, help='Predict x0')
    parser.add_argument('--lower_order_final', type=bool, default=True, help='Lower order at final steps')
    parser.add_argument('--use_fp16', action='store_true', help='Use FP16 precision for training')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (optional)')
    parser.add_argument('--nosubdir', action='store_true', help='Do not use subdirectory for run')
    parser.add_argument('--outdir', default='./exps', help='Output directory')
    parser.add_argument('--desc', default=None, help='Custom description string')
    parser.add_argument('--dry_run', action='store_true', help='Perform a dry run without training')
    
    args = parser.parse_args()
    return args

def main():
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    args = parse_args()
    
    c = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.AMED_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=args.lr, betas=[0.9, 0.999], eps=1e-8)

    c.AMED_kwargs.class_name = 'training_amed.networks.AMED_predictor'
    c.AMED_kwargs.update(num_steps=args.num_steps, sampler_stu=args.sampler_stu, sampler_tea=args.sampler_tea,
                         M=args.M, guidance_type=args.guidance_type, guidance_rate=args.guidance_rate,
                         schedule_rho=args.schedule_rho, schedule_type=args.schedule_type, afs=args.afs,
                         dataset_name=args.dataset_name, scale_dir=args.scale_dir, scale_time=args.scale_time,
                         max_order=args.max_order, predict_x0=args.predict_x0, lower_order_final=args.lower_order_final)
    c.loss_kwargs.class_name = 'training_amed.loss.AMED_loss'

    c.total_kimg = args.total_kimg
    c.kimg_per_tick = 1
    c.snapshot_ticks = args.total_kimg
    c.state_dump_ticks = args.total_kimg
    c.update(dataset_name=args.dataset_name, batch_size=args.batch, batch_gpu=args.batch_gpu, gpus=dist.get_world_size(),
             cudnn_benchmark=args.bench, use_fp16=args.use_fp16,image_size=args.image_size,checkpoint_path=args.checkpoint_path)
    c.update(guidance_type=args.guidance_type, guidance_rate=args.guidance_rate, prompt_path=args.prompt_path)

    if args.seed is not None:
        c.seed = args.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    if args.schedule_type == 'polynomial':
        schedule_str = 'poly' + str(args.schedule_rho)
    elif args.schedule_type == 'logsnr':
        schedule_str = 'logsnr'
    elif args.schedule_type == 'time_uniform':
        schedule_str = 'uni' + str(args.schedule_rho)
    elif args.schedule_type == 'discrete':
        schedule_str = 'discrete'
    else:
        raise ValueError(f"Got wrong schedule type: {args.schedule_type}")
    nfe = 2 * (args.num_steps - 1) - 1 if args.afs else 2 * (args.num_steps - 1)
    nfe = 2 * nfe if args.dataset_name == 'ms_coco' else nfe
    if args.afs:
        desc_str = f'{args.dataset_name}-{args.num_steps}-{nfe}-{args.sampler_stu}-{args.sampler_tea}-{args.M}-{schedule_str}-afs'
    else:
        desc_str = f'{args.dataset_name}-{args.num_steps}-{nfe}-{args.sampler_stu}-{args.sampler_tea}-{args.M}-{schedule_str}'
    if args.desc is not None:
        desc_str += f'-{args.desc}'
    c.desc = desc_str

    if dist.get_rank() != 0:
        c.run_dir = None
    elif args.nosubdir:
        c.run_dir = args.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(args.outdir):
            prev_run_dirs = [x for x in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(args.outdir, f'{cur_run_id:05d}-{desc_str}')
        assert not os.path.exists(c.run_dir)

    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0()

    if args.dry_run:
        dist.print0('Dry run; exiting.')
        return

    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    training_loop.training_loop(**c)

if __name__ == "__main__":
    main()
