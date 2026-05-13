import torch
import numpy as np

from losses.diffaug import DiffAugment as aug
from torchvision.transforms import Normalize

import math

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    return torch.sum(x, dim=list(range(1, len(x.size()))))


def info_per_layer(N, Imin=0.125):
    idx = np.arange(N)
    lam = idx / (N - 1)           
    return 1. - Imin ** (1 - lam)

class RpGANPTLoss:
    def __init__(
            self,
            encoders=[], 
            encoder_types=[],
            architectures=[],
            accelerator=None,
            r1_gamma=0.1,
            r1_every=8,
            r2_gamma=0.1,
            r2_every=8,
            proj_coeff=1.0,
            approximate=False,
            ):
        self.encoders = encoders
        self.encoder_types = encoder_types
        self.architectures = architectures
        self.accelerator = accelerator
        
        self.proj_coeff = proj_coeff
        
        self.r1_gamma = r1_gamma
        self.r2_gamma = r2_gamma
        self.r1_every = r1_every
        self.r2_every = r2_every
        
        self.approximate = approximate
        self.policy = 'color,translation,cutout,flip'
        self.policy_raw_image = 'translation,flip'
        
        self.approximated_GP_std = 0.01
        self.aug_prob = 1.0
        
    def apply_gaussian(self, x, t, seed_noise=None):
        if seed_noise is None:
            seed_noise = torch.randn_like(x)
            
        return x * (1. - t) + seed_noise * t, seed_noise
    
    def apply_gaussian_list(self, xs, seed_noise=None, min_info=0.125):
        noise_schedule = info_per_layer(N=4, Imin=min_info)
        
        noise_schedule = noise_schedule + 1e-2
        
        n_ts = len(xs)
        xs_new = []
        noises = []
        
        if seed_noise is None:
            seed_noise = [None for _ in range(n_ts)]
        seed_noise = seed_noise[-n_ts:]
        noise_schedule = noise_schedule[-n_ts:]
        
        for i in range(n_ts):
            x_noised, seed_noise_ = self.apply_gaussian(xs[i], noise_schedule[i], seed_noise[i])
            xs_new.append(x_noised) 
            noises.append(seed_noise_)
            
        return torch.stack(xs_new, dim=0), torch.stack(noises, dim=0)
    
    def apply_gaussian_list_cumulative(self, xs, seed_noise=None, min_info=0.125):
        noise_schedule = info_per_layer(N=4, Imin=min_info)
        
        noise_schedule = noise_schedule + 1e-2
        
        n_ts = len(xs)
        xs_new = []
        
        if seed_noise is None:
            seed_noise = [torch.randn_like(xs[_]) for _ in range(n_ts)]
        
        cumulative_noise = torch.zeros_like(xs[0])
        
        prev_t_squared, prev_t = 0.0, 0.0

        noise_schedule = list(noise_schedule)[::-1]
        xs = list(xs)[::-1]
        
        for idx, t in enumerate(noise_schedule):

            current_t_squared = t**2
            
            decay_ratio = (1. - t) / (1. - prev_t)
            
            incremental_variance = current_t_squared - prev_t_squared * decay_ratio ** 2

            incremental_noise_sample = seed_noise[idx]
            
            scaled_incremental_noise = incremental_noise_sample * math.sqrt(incremental_variance)
            
            cumulative_noise = cumulative_noise * decay_ratio + scaled_incremental_noise
            
            x_noisy = (1 - t) * xs[idx] + cumulative_noise
            
            xs_new.append(x_noisy)
            
            prev_t_squared = current_t_squared
            prev_t = t

        xs_new = list(xs_new)[::-1]
        return torch.stack(xs_new, dim=0), seed_noise
    
    def approximated_gradient_penalty(self, data, pred, model, model_kwargs={}):

        aug_params, noise_params = model_kwargs["aug_params"], model_kwargs["noise_params"]
        
        if len(data.shape) > 4:
            
            data_list = []
            for idx, d in enumerate(data):
                d, _ = aug(d + torch.randn_like(d) * self.approximated_GP_std, aug_params=aug_params, policy=self.policy)
                data_list.append(d)
            data = torch.stack(data_list, dim=0)
        else:
            data, _ = aug(data + torch.randn_like(data) * self.approximated_GP_std, aug_params=aug_params, policy=self.policy)
            data = data.unsqueeze(0).repeat(len(noise_params), 1, 1, 1, 1)  
            
        data, _ = self.apply_gaussian_list_cumulative(data, noise_params)
        
        pred_noised = model(data, y=model_kwargs["y"])
        
        return ((pred - pred_noised) / self.approximated_GP_std).pow(2).mean(-1)
    
    
    def gradient_penalty(self, data, pred, model_kwargs={}):
        gradients = torch.autograd.grad(
            outputs=pred.sum(), inputs=data,
            create_graph=True, retain_graph=True)[0]
        gradient_penalty = gradients.pow(2).sum([1, 2, 3]).mean()
        return gradient_penalty
    
    
    def encode_feature(self, raw_image, do_aug=True, aug_params=None):
        zs = []
        with self.accelerator.autocast():
            with torch.no_grad():
                for encoder, encoder_type, arch in zip(self.encoders, self.encoder_types, self.architectures):
                    raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                    
                    if do_aug:
                        raw_image_, _ = aug(raw_image_, aug_params=aug_params, policy=self.policy_raw_image)
                    
                    z = encoder.forward_features(raw_image_)
                    
                    assert 'dinov2' in encoder_type
                    
                    zs.append(z)
        return zs
    
    def step_gen(self, generator, discriminator, discriminator_ema, images, raw_images, global_step, model_kwargs=None, zs=None, **kwargs):

        if model_kwargs == None:
            model_kwargs = {}
        if "z" not in model_kwargs.keys():
            z = torch.randn(images.shape[0], generator.module.hidden_size, device=images.device, dtype=images.dtype)
            model_kwargs["z"] = z
        if "x" not in model_kwargs.keys():
            model_kwargs["x"] = torch.randn_like(images)
        
        
        gen_images  = generator(update_ema=True, multiscale=True, **model_kwargs)
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_aug_params = aug(gen_images[-1], prob=self.aug_prob, policy=self.policy)
            
            gen_images_aug_list = []
            for gen_img in gen_images[:-1]:
                gen_images_aug_list.append(aug(gen_img, aug_params=gen_aug_params, policy=self.policy)[0])
            gen_images_aug = torch.stack(gen_images_aug_list + [gen_images_aug], dim=0)
            
            images_aug, real_aug_params = aug(images.detach(), prob=self.aug_prob, policy=self.policy)
        else:    
            gen_images_aug, gen_aug_params = aug(gen_images, prob=self.aug_prob, policy=self.policy)
            images_aug, real_aug_params = aug(images.detach(), prob=self.aug_prob, policy=self.policy)
        
        images_aug = images_aug.unsqueeze(0).repeat(len(gen_images_aug), 1, 1, 1, 1)
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_noise = self.apply_gaussian_list_cumulative(gen_images_aug)
            images_aug, real_noise = self.apply_gaussian_list_cumulative(images_aug)
            
        
        gen_logits, aux = discriminator(gen_images_aug, y=model_kwargs["y"], return_aux=True)
        
        zs = aux["x_feat"]
        with torch.no_grad():
            real_logits = discriminator(images_aug, y=model_kwargs["y"])
        
        
        relativistic_logits = gen_logits - real_logits
        gen_loss = torch.nn.functional.softplus(-relativistic_logits).mean(-1)
        
        z_loss = zs[0].mean() * 0.0 + zs[1].mean() * 0.0
        loss = gen_loss + z_loss
        
        loss_dict = {
            "gen_loss": gen_loss,
        }
        extras = {
            "gen_images": gen_images
        }
        return loss, loss_dict, extras
    

    def step_disc(self, generator, discriminator, discriminator_ema, images, raw_images, global_step, model_kwargs=None, zs=None, **kwargs):

        r1_gamma, r2_gamma = self.r1_gamma, self.r2_gamma
        aug_prob = 1.0
        
        
        if model_kwargs == None:
            model_kwargs = {}
        if "z" not in model_kwargs.keys():
            z = torch.randn(images.shape[0], generator.module.latent_size, device=images.device, dtype=images.dtype)
            model_kwargs["z"] = z
        if "x" not in model_kwargs.keys():
            model_kwargs["x"] = torch.randn_like(images)
            
                                            
        with torch.no_grad():
            gen_images  = generator(update_ema=False, multiscale=True, **model_kwargs)
        
        
        if global_step % self.r1_every == 0:
            images = images.detach()
            images.requires_grad = True
        if global_step % self.r2_every == 0:
            gen_images = gen_images.detach()
            gen_images.requires_grad = True
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_aug_params = aug(gen_images[-1], prob=self.aug_prob, policy=self.policy)
            
            gen_images_aug_list = []
            for gen_img in gen_images[:-1]:
                gen_images_aug_list.append(aug(gen_img, aug_params=gen_aug_params, policy=self.policy)[0])
            gen_images_aug = torch.stack(gen_images_aug_list + [gen_images_aug], dim=0)
            
            images_aug, real_aug_params = aug(images, prob=self.aug_prob, policy=self.policy)
        else:    
            gen_images_aug, gen_aug_params = aug(gen_images, prob=self.aug_prob, policy=self.policy)
            images_aug, real_aug_params = aug(images, prob=self.aug_prob, policy=self.policy)
        
        images_aug = images_aug.unsqueeze(0).repeat(len(gen_images_aug), 1, 1, 1, 1)
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_noise = self.apply_gaussian_list_cumulative(gen_images_aug)
            images_aug, real_noise = self.apply_gaussian_list_cumulative(images_aug)
        
        gen_logits, gen_aux = discriminator(gen_images_aug, y=model_kwargs["y"], return_aux=True)
        real_logits, real_aux = discriminator(images_aug, y=model_kwargs["y"], return_aux=True)
        
        z_feat_tilde = real_aux["x_feat"]
        
        with torch.no_grad():
            z_feat = self.encode_feature(raw_images, do_aug=True, aug_params=real_aug_params)[0]
        
        z_cls_kd_loss = 1. - torch.nn.functional.cosine_similarity(z_feat_tilde[0].squeeze(1), z_feat['x_norm_clstoken'].detach(), dim=-1)
        
        
        z_feat_spatial = z_feat['x_norm_patchtokens']
        
        
        if z_feat_tilde[1].shape != z_feat_spatial.shape:
            z_feat_spatial = resize_spatial(z_feat_spatial, H_out=int(np.sqrt(z_feat_tilde[1].shape[1])))
        
        
        z_spatial_kd_loss = (1. - torch.nn.functional.cosine_similarity(z_feat_tilde[1], z_feat_spatial.detach(), dim=-1)).sum(dim=1) / z_feat_spatial.shape[1]
        
        z_kd_loss = z_cls_kd_loss + z_spatial_kd_loss
        
        relativistic_logits_cls = real_logits - gen_logits
        disc_loss = torch.nn.functional.softplus(-relativistic_logits_cls).mean(-1)
        
        loss = disc_loss + z_kd_loss * self.proj_coeff
        loss_dict = {
            "disc_loss": disc_loss,
            "z_kd_cls": z_cls_kd_loss,
            "z_kd_patch": z_spatial_kd_loss,
        }
        
        assert len(real_logits.shape) == 2, "logit should be [B, nlogits] for current multi-scale pred implementation"
        if global_step % self.r1_every == 0:
            if self.approximate:
                model_kwargs_GP = {"y": model_kwargs["y"], "aug_params": real_aug_params, "noise_params": real_noise}
                r1_loss = self.approximated_gradient_penalty(images, real_logits, discriminator, model_kwargs_GP)
            else:
                r1_loss = self.gradient_penalty(images, real_logits.mean(-1), discriminator, model_kwargs)
            loss += r1_gamma / 2 * r1_loss
            loss_dict["r1_loss"] = r1_loss
        if global_step % self.r2_every == 0:
            if self.approximate:
                model_kwargs_GP = {"y": model_kwargs["y"], "aug_params": gen_aug_params, "noise_params": gen_noise}
                r2_loss = self.approximated_gradient_penalty(gen_images, gen_logits, discriminator, model_kwargs_GP)
            else:
                r2_loss = self.gradient_penalty(gen_images, gen_logits.mean(-1), discriminator, model_kwargs)
            loss += r2_gamma / 2 * r2_loss
            loss_dict["r2_loss"] = r2_loss
            
        extras = {}
        return loss, loss_dict, extras
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')

    return x


def resize_spatial(tensor: torch.Tensor, H_out: int, W_out: int = None):
    B, L, C = tensor.shape
    H_in = W_in = int(np.sqrt(L))
    assert H_in * W_in == L, f"L={L} cannot be reshaped to square"

    if W_out is None:
        W_out = H_out
    
    x = tensor.transpose(1, 2).reshape(B, C, H_in, W_in)
    x = torch.nn.functional.interpolate(x, size=(H_out, W_out), mode='bicubic', align_corners=False)
    x = x.reshape(B, C, -1).transpose(1, 2)
    
    return x
