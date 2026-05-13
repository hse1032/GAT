import torch
import numpy as np

from losses.diffaug import DiffAugment as aug

import math

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    return torch.sum(x, dim=list(range(1, len(x.size()))))


def info_per_layer(N, Imin=0.125):
    idx = np.arange(N)
    lam = idx / (N - 1)           
    return 1. - Imin ** (1 - lam)

class RpGANLoss:
    def __init__(
            self,
            encoders=[], 
            accelerator=None,
            r1_gamma=0.1,
            r1_every=8,
            r2_gamma=0.1,
            r2_every=8,
            approximate=False,
            ):
        self.encoders = encoders
        self.accelerator = accelerator
        
        
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
    
    def step_gen(self, generator, discriminator, discriminator_ema, images, raw_images, global_step, model_kwargs=None, zs=None, **kwargs):

        if model_kwargs == None:
            model_kwargs = {}
        if "z" not in model_kwargs.keys():
            z = torch.randn(images.shape[0], generator.module.latent_size, device=images.device, dtype=images.dtype)
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
            
        
        gen_logits = discriminator(gen_images_aug, y=model_kwargs["y"])
        
        with torch.no_grad():
            real_logits = discriminator(images_aug, y=model_kwargs["y"])
        
        relativistic_logits = gen_logits - real_logits
        gen_loss = torch.nn.functional.softplus(-relativistic_logits).mean(-1)
        
        loss = gen_loss
        
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
        
        gen_logits = discriminator(gen_images_aug, y=model_kwargs["y"])
        real_logits = discriminator(images_aug, y=model_kwargs["y"])
        
        relativistic_logits_cls = real_logits - gen_logits
        disc_loss = torch.nn.functional.softplus(-relativistic_logits_cls).mean(-1)
        
        loss = disc_loss
        loss_dict = {
            "disc_loss": disc_loss,
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
