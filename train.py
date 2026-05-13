import argparse
import copy
from copy import deepcopy
import logging
import os
import shutil
from pathlib import Path
from collections import OrderedDict
import json

import torch
import torch.utils.checkpoint

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs

from models.generator import GAT_models
from models.discriminator import GATD_models
from losses import RpGANLoss, RpGANPTLoss
from utils import load_encoders

from dataset import CustomDataset, CustomDataset_DiT
from diffusers.models import AutoencoderKL

import wandb
import math
from torchvision.utils import make_grid

from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

logger = get_logger(__name__)

def normalize_model_name(name):
    return name.replace("SiT-", "GAT-", 1) if name.startswith("SiT-") else name


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    ema_buffers = OrderedDict(ema_model.named_buffers())  
    model_buffers = OrderedDict(model.named_buffers())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if name in ema_buffers:
            ema_buffers[name].copy_(buffer)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def main(args):    
    args.model = normalize_model_name(args.model)
    args.modelD = normalize_model_name(args.modelD)
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False), InitProcessGroupKwargs(timeout=timedelta(seconds=5400))],
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
        
        if args.resume_step == 0:
            os.makedirs(os.path.join(save_dir, "source_codes"), exist_ok=True)
            shutil.copytree(os.getcwd(), os.path.join(save_dir, "source_codes"), dirs_exist_ok=True, \
                            ignore=shutil.ignore_patterns('wandb', '_experiments', 'features_temp', 'temp', 'pretrained_models'))
        
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != "None":
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
    else:
        print("Do not load pretrained network, so MAE loss is only used.")
        
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    generator = GAT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        z_dims = z_dims,
        **block_kwargs
    )
    discriminator = GATD_models[args.modelD](
        input_size=latent_size,
        num_classes=args.num_classes,
        z_dims = z_dims,
        **block_kwargs
    )

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    ema = deepcopy(generator).to(device)
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    if args.enc_type == 'None':
        loss_fn = RpGANLoss(
            encoders=None,
            accelerator=accelerator,
            r1_every=args.R1_every,
            r1_gamma=args.R1_gamma,
            r2_every=args.R2_every,
            r2_gamma=args.R2_gamma,
            approximate=True,
        )
    else:
        loss_fn = RpGANPTLoss(
            encoders=encoders,
            encoder_types=encoder_types,
            architectures=architectures,
            accelerator=accelerator,
            r1_every=args.R1_every,
            r1_gamma=args.R1_gamma,
            r2_every=args.R2_every,
            r2_gamma=args.R2_gamma,
            proj_coeff=args.proj_coeff,
            approximate=True,
        )
    
    if accelerator.is_main_process:
        logger.info(f"GAT Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True

    glr = args.learning_rate * 768 / generator.hidden_size
    dlr = args.learning_rate * 768 / discriminator.hidden_size

    optimizerG = torch.optim.AdamW(
        generator.parameters(),
        lr=glr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    optimizerD = torch.optim.AdamW(
        discriminator.parameters(),
        lr=dlr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    

    if args.resolution == 256:
        train_dataset = CustomDataset(args.data_dir)
    elif args.resolution == 128:
        train_dataset = CustomDataset_DiT(features_dir="../dataset_sit/imagenet_features_128/imagenet256_features", \
                                        labels_dir="../dataset_sit/imagenet_features_128/imagenet256_labels")
    else:
        raise ValueError(f"Unsupported resolution: {args.resolution}")
    
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    global_step = 0
    wallclock_time = 0.
    fid_best = 1e+5
    fid_cur = 1e+5
    
    if args.resume_step != 0:
        if args.resume_step < 0:
            ckpt_name = 'latest.pt'
        else:
            ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            weights_only=False,
        )
        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        ema.load_state_dict(ckpt['ema'])
        optimizerG.load_state_dict(ckpt['optG'])
        optimizerD.load_state_dict(ckpt['optD'])

        global_step = ckpt['steps']
        wallclock_time = ckpt['wallclock_time']
        wandb_run_id = None
        is_resume = "allow"
    else:
        wandb_run_id = None
        is_resume = None

    generator, discriminator, optimizerG, optimizerD, train_dataloader = accelerator.prepare(
        generator, discriminator, optimizerG, optimizerD, train_dataloader
    )
    
    update_ema(ema, generator, decay=0)

    generator.train()
    discriminator.train()
    ema.eval()
    
    
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name=args.wandb_name,
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    "dir": save_dir,
                    "name": f"{args.exp_name}",
                    "id": wandb_run_id,
                    "resume": is_resume,
                    }
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    sample_batch_size = 64 // accelerator.num_processes
    
    ys_vis = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279], device=device).repeat(sample_batch_size // 8)
    ys_vis = ys_vis.to(device)
    n = ys_vis.size(0)
    zs_vis = torch.randn(size=(sample_batch_size, generator.module.latent_size), device=device)
    xs_vis = torch.randn((n, 4, latent_size, latent_size), device=device)
    stats_metrics = dict()
    
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        
        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)

            with torch.no_grad():
                if x.shape[1] == 8:
                    x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)

            model_kwargs = dict(y=y)
            with accelerator.accumulate(discriminator):
                discriminator.module.requires_grad_(True)
                loss, loss_dict, _ = loss_fn.step_disc(generator, discriminator, None, x, raw_image, global_step, model_kwargs)
                loss = loss.mean()
                    
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = discriminator.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizerD.step()
                optimizerD.zero_grad(set_to_none=True)
                    
                discriminator.module.requires_grad_(False)

            with accelerator.accumulate(generator):
                generator.module.requires_grad_(True)
                loss, _, extras = loss_fn.step_gen(generator, discriminator, None, x, raw_image, global_step, model_kwargs)

                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = generator.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizerG.step()
                optimizerG.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    update_ema(ema, generator)
                generator.module.requires_grad_(False)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            
            if (global_step % args.eval_steps == 0) or (args.resume_step == global_step - 1):
                from metrics import metric_main

                if accelerator.is_main_process:
                    print('Evaluating metrics...')
                for metric in ["fid5k"]:
                    with torch.no_grad():
                        result_dict = metric_main.calc_metric(metric=metric, G=ema, vae=vae, latent_bias=latents_bias, latent_scale=latents_scale, \
                                            accelerator=accelerator, real_npy=os.path.join(args.data_dir, f"VIRTUAL_imagenet{args.resolution}_labeled.npz"))
                        if accelerator.process_index == 0:
                            metric_main.report_metric(result_dict, run_dir=save_dir, snapshot_pkl=args.exp_name)
                        stats_metrics.update(result_dict.results)
                    
                logs = {}
                for name, value in stats_metrics.items():
                    logs[name] = value
                accelerator.log(logs, step=global_step)
                fid_cur = logs["fid5k"]
                
            if ((global_step % args.checkpointing_steps == 0) or (global_step % args.latest_checkpointing_steps == 0)) and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "generator": generator.module.state_dict(),
                        "discriminator": discriminator.module.state_dict(),
                        "ema": ema.state_dict(),
                        "optG": optimizerG.state_dict(),
                        "optD": optimizerD.state_dict(),
                        "args": args,
                        "steps": global_step,
                        "wallclock_time": progress_bar.format_dict['elapsed'],
                        "wandb_run_id": accelerator.get_tracker("wandb").run.id,
                    }
                    if global_step % args.checkpointing_steps == 0:
                        checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if global_step % args.latest_checkpointing_steps == 0:
                        checkpoint_path = f"{checkpoint_dir}/latest.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if fid_best > fid_cur:
                        checkpoint_path = f"{checkpoint_dir}/fid_best.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Update best FID-5K checkpoint to {checkpoint_path}")
                        fid_best = fid_cur
                         

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                with torch.no_grad():
                    samples = ema(x=xs_vis, z=zs_vis, y=ys_vis, truncation_psi=0.5)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    
                    samples_notruc = ema(x=xs_vis, z=zs_vis, y=ys_vis, truncation_psi=0.0)
                    samples_notruc = vae.decode((samples_notruc -  latents_bias) / latents_scale).sample
                    samples_notruc = (samples_notruc + 1) / 2.
                    
                out_samples = accelerator.gather(samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples))})
                
                out_samples_notruc = accelerator.gather(samples_notruc.to(torch.float32))
                accelerator.log({"samples w/o trunc": wandb.Image(array2grid(out_samples_notruc))})
                
                logging.info("Generating EMA samples done.")

            logs = {}
            for k in loss_dict.keys():
                logs[k] = accelerator.gather(loss_dict[k].mean()).mean().detach().item()

            logs["x_last_std"] = accelerator.gather(generator.module.recent_x_std.mean()).mean().detach().item()
            logs["x_last_std_disc"] = accelerator.gather(discriminator.module.recent_x_std.mean()).mean().detach().item()
            
            progress_bar.set_postfix(**logs)

            logs["wallclock_time"] = progress_bar.format_dict['elapsed']
            
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    generator.eval()
    discriminator.eval()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=1250)
    parser.add_argument("--eval-steps", type=int, default=2500)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--wandb-name", type=str, default="GAN 2025")

    parser.add_argument("--model", type=str)
    parser.add_argument("--modelD", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--R1_gamma",  type=float, default=1e-1)
    parser.add_argument("--R2_gamma",  type=float, default=1e-1)
    parser.add_argument("--R1_every",  type=int, default=1)
    parser.add_argument("--R2_every",  type=int, default=1)

    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=20000)
    parser.add_argument("--latest-checkpointing-steps", type=int, default=1250)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.0, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.99, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num-workers", type=int, default=16)

    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=1.0)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    import os
    os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
    from torch._inductor import config as ind
    ind.triton.cudagraphs = False

    main(args)
