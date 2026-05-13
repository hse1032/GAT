import argparse
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm

from models.generator import GAT_models
from utils import load_legacy_checkpoints


def normalize_model_name(name):
    return name.replace("SiT-", "GAT-", 1) if name.startswith("SiT-") else name


def get_checkpoint_state(checkpoint, weight_key):
    for key in (weight_key, "ema", "generator", "model"):
        state_dict = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if isinstance(state_dict, dict):
            return state_dict, key
    raise RuntimeError("Checkpoint does not contain model weights.")


def apply_checkpoint_args(args, checkpoint):
    ckpt_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if ckpt_args is None:
        return args
    for name in ("model", "resolution", "num_classes", "fused_attn", "qk_norm"):
        if hasattr(ckpt_args, name):
            setattr(args, name, getattr(ckpt_args, name))
    args.model = normalize_model_name(args.model)
    return args


def create_npz_from_sample_folder(sample_dir, num):
    samples = []
    for i in tqdm(range(num), desc="Building npz"):
        samples.append(np.asarray(Image.open(f"{sample_dir}/{i:06d}.png")).astype(np.uint8))
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=np.stack(samples))
    print(f"Saved {npz_path}.")


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling requires CUDA.")
    if args.ckpt is None:
        raise ValueError("--ckpt is required.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed * world_size + rank)

    checkpoint = torch.load(args.ckpt, weights_only=False, map_location=f"cuda:{device}")
    args = apply_checkpoint_args(args, checkpoint)
    state_dict, state_key = get_checkpoint_state(checkpoint, args.weight_key)
    if args.legacy:
        state_dict = load_legacy_checkpoints(state_dict, encoder_depth=args.encoder_depth)

    latent_size = args.resolution // 8
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = GAT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(",")],
        **block_kwargs,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    latents_scale = torch.tensor([0.18215] * 4, device=device).view(1, 4, 1, 1)
    latents_bias = torch.zeros(1, 4, 1, 1, device=device)

    if rank == 0:
        print(f"Loaded {state_key}: model={args.model}, resolution={args.resolution}")
        print(f"Generator parameters: {sum(p.numel() for p in model.parameters()):,}")

    folder_name = (
        f"{args.model.replace('/', '-')}-{os.path.basename(args.ckpt).replace('.pt', '')}"
        f"-size-{args.resolution}-vae-{args.vae}-seed-{args.global_seed}"
    )
    sample_folder = os.path.join(args.sample_dir, folder_name)
    if rank == 0:
        os.makedirs(sample_folder, exist_ok=True)
    dist.barrier()

    per_rank_batch = args.per_proc_batch_size
    global_batch = per_rank_batch * world_size
    total_samples = int(math.ceil(args.num_fid_samples / global_batch) * global_batch)
    iterations = total_samples // global_batch
    total = 0
    pbar = tqdm(range(iterations), disable=(rank != 0))

    for _ in pbar:
        x = torch.randn(per_rank_batch, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (per_rank_batch,), device=device)
        z = torch.randn(per_rank_batch, model.latent_size, device=device)
        latents = model(x=x, y=y, z=z, truncation_psi=args.truncation_psi)
        images = vae.decode((latents - latents_bias) / latents_scale).sample
        images = (images + 1) / 2
        images = torch.clamp(255 * images, 0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

        for i, image in enumerate(images):
            index = total + i * world_size + rank
            if index < args.num_fid_samples:
                Image.fromarray(image).save(f"{sample_folder}/{index:06d}.png")
        total += global_batch

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder, args.num_fid_samples)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--weight-key", type=str, choices=["ema", "generator", "model"], default="ema")
    parser.add_argument("--model", type=str, choices=list(GAT_models.keys()), default="GAT-S/4")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--projector-embed-dims", type=str, default="768,1024")
    parser.add_argument("--truncation-psi", type=float, default=0.3)
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    main(parser.parse_args())
