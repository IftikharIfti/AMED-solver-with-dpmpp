"""Main training loop."""
import argparse
import os
import csv
import time
import copy
import json
import pickle
import numpy as np
import torch.nn.functional as F
import torch
import dnnlib
import random
from torch import autocast
from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion, args_to_dict, model_and_diffusion_defaults
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from ldm.util import instantiate_from_config
from torch_utils.download_util import check_file_by_key

def training_loop(
    run_dir             = '.',      # Output directory.
    AMED_kwargs         = {},       # Options for AMED predictor.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    seed                = 0,        # Global random seed.
    batch_size          = None,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 20,       # Training duration, measured in thousands of training images.
    kimg_per_tick       = 1,        # Interval of progress prints.
    snapshot_ticks      = 1,        # How often to save network snapshots, None = disable.
    state_dump_ticks    = 20,       # How often to dump training state, None = disable.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    dataset_name        = None,
    prompt_path         = None,
    guidance_type       = None,
    guidance_rate       = 0.,
    device              = torch.device('cuda'),
    image_size          = 64,       # Add image_size as a parameter with default value
    **kwargs,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Custom M2S LWDM defaults overriding model_and_diffusion_defaults()
    custom_defaults = model_and_diffusion_defaults()
    custom_defaults.update({
        'attention_resolutions': "16",
        'class_cond': False,
        'diffusion_steps': 1000,
        'dropout': 0.0,
        'learn_sigma': True,
        'noise_schedule': "linear",
        'num_channels': 128,
        'num_head_channels': 64,
        'num_res_blocks': 1,
        'resblock_updown': True,
        'use_fp16': True,
        'use_scale_shift_norm': True,
        'rescale_learned_sigmas': True,
        'p2_gamma': 1,
        'p2_k': 1,
        'image_size': image_size,  # Set image_size from args
    })
    # Load LWDM model based on args.image_size
    #custom_defaults['image_size'] = args.image_size  # Set from command-line argument
    args = argparse.Namespace(**custom_defaults)
    dist.print0(f'Loading LWDM model ({args.image_size}x{args.image_size})...')
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, custom_defaults.keys()))
    checkpoint = dist_util.load_state_dict(kwargs["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict'])
    if args.use_fp16:
        model.convert_to_fp16()
    model.to(device)
    
    # Construct AMED predictor
    dist.print0('Constructing AMED predictor...')
    AMED_kwargs.update(img_resolution=args.image_size)  # Set resolution dynamically
    predictor = dnnlib.util.construct_class_by_name(**AMED_kwargs).to(device)
    predictor.train().requires_grad_(True)
    
    # Setup optimizer
    dist.print0('Setting up optimizer...')
    loss_kwargs.update(
        num_steps=AMED_kwargs['num_steps'],
        sampler_stu=AMED_kwargs['sampler_stu'],
        sampler_tea=AMED_kwargs['sampler_tea'],
        M=AMED_kwargs['M'],
        schedule_type=AMED_kwargs['schedule_type'],
        schedule_rho=AMED_kwargs['schedule_rho'],
        afs=AMED_kwargs['afs'],
        max_order=AMED_kwargs['max_order'],
        sigma_min=0.002,
        sigma_max=80,
        predict_x0=AMED_kwargs['predict_x0'],
        lower_order_final=AMED_kwargs['lower_order_final']
    )
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    
    optimizer = dnnlib.util.construct_class_by_name(params=predictor.parameters(), **optimizer_kwargs)
    ddp = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[device], broadcast_buffers=False)

    # Train
    dist.print0(f'Training for {total_kimg} kimg...')
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:
        # Generate latents based on image size
        latents = diffusion.q_sample(
            torch.randn([batch_gpu, 3, args.image_size, args.image_size], device=device),
            torch.tensor([diffusion.num_timesteps - 1], device=device)
        )
    
        optimizer.zero_grad()
        teacher_traj = loss_fn.get_teacher_traj(net=model, tensor_in=latents,diffusion=diffusion)
        for step_idx in range(loss_fn.num_steps - 1):
            loss, stu_out = loss_fn(
                AMED_predictor=ddp,
                net=model,
                tensor_in=latents,
                step_idx=step_idx,
                teacher_out=teacher_traj[step_idx],
                diffusion=diffusion
            )
            loss.sum().mul(1 / batch_gpu_total).backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            latents = stu_out if AMED_kwargs['sampler_stu'] not in ['euler', 'dpm', 'amed'] else teacher_traj[step_idx]
    
        cur_nimg += batch_size
        if cur_nimg >= total_kimg * 1000 or (cur_tick % snapshot_ticks == 0 and cur_tick > 0):
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'predictor_{args.image_size}-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump({'model': predictor.cpu()}, f)
            if cur_nimg >= total_kimg * 1000:
                break
            cur_tick += 1
    
    dist.print0('Exiting...')
