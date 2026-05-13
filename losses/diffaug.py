# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F
import numpy as np

def DiffAugment(x, prob=1.0, policy='', channels_first=True, aug_params=None):
    if np.random.rand() > prob:
        return x, {}
    
    aug_params_new = {}
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)

        for p in policy.split(','):
            
            aug_params_new[p] = []
            
            if aug_params is None:
                for f in AUGMENT_FNS[p]:
                    x, _param = f(x)
                    aug_params_new[p].append(_param)
            else:
                for f, _param in zip(AUGMENT_FNS[p], aug_params[p]):
                    x, _param = f(x, _param)
                    aug_params_new[p].append(_param)
            
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x, aug_params_new


def rand_brightness(x, params=None):
    if params is None:
        noise = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = x + (noise - 0.5)
        params = noise
    else:
        noise = params
        x = x + (noise - 0.5)
    return x, params


def rand_saturation(x, params=None):
    if params is None:
        x_mean, noise = x.mean(dim=1, keepdim=True), torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (noise * 2) + x_mean
        params = (x_mean, noise)
    else:
        x_mean, noise = params
        x = (x - x_mean) * (noise * 2) + x_mean
    return x, params


def rand_contrast(x, params=None):
    if params is None:
        x_mean, noise = x.mean(dim=[1, 2, 3], keepdim=True), torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (noise + 0.5) + x_mean
        params = (x_mean, noise)
    else:
        x_mean, noise = params
        x = (x - x_mean) * (noise + 0.5) + x_mean
    return x, params


def rand_brightness_saturation_contrast(x, params=None, p=0.5):
    
    if params is None:
        noise_bright = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x_mean_sat, noise_sat = x.mean(dim=1, keepdim=True), torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x_mean_cont, noise_cont = x.mean(dim=[1, 2, 3], keepdim=True), torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    
        mask = torch.rand(x.size(0), device=x.device)
        mask = mask < p
    else:
        mask, noise_bright, x_mean_sat, noise_sat, x_mean_cont, noise_cont = params
        
    x_aug = x + (noise_bright - 0.5)
    x_aug = (x_aug - x_mean_sat) * (noise_sat * 2) + x_mean_sat
    x_aug = (x_aug - x_mean_cont) * (noise_cont + 0.5) + x_mean_cont

    x_out = x.clone()
    x_out[mask] = x_aug[mask]
    
    params = [mask, noise_bright, x_mean_sat, noise_sat, x_mean_cont, noise_cont]

    return x_out, params
    


def rand_translation(x, params=None, ratio=0.125):
    if params is None:
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        params = (translation_x, translation_y, x.shape[2], x.shape[3])
    else:
        translation_x, translation_y, h, w = params
        translation_x, translation_y = translation_x * h // x.shape[2], translation_y * w // x.shape[3]
        
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x, params


def rand_cutout(x, params=None, ratio=0.5):
    if params is None:
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        params = (cutout_size, offset_x, offset_y, x.shape[2], x.shape[3])
    else:
        cutout_size, offset_x, offset_y, h, w = params
        cutout_size, offset_x, offset_y = (cutout_size[0] * h // x.shape[2], cutout_size[1] * w // x.shape[3]), offset_x * h // x.shape[2], offset_y * w // x.shape[3]
        
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x, params


def rand_flip(x: torch.Tensor,
                    params: torch.Tensor | None = None,
                    ratio: float = 0.4,
                    dim: int = 3):
    B = x.size(0)

    if params is None:
        params = torch.rand(B, device=x.device)

    mask = params < ratio
    if mask.any():
        x_out = x.clone()
        x_out[mask] = x[mask].flip(dim)
        return x_out, params
    else:
        return x, params

AUGMENT_FNS = {
    'color': [rand_brightness_saturation_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
}
