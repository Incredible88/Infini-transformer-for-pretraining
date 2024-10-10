#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : MetaIndex 
# @File    : gaussian_diffusion.py
# @Author  : zhangchao
# @Date    : 2024/7/23 10:53 
# @Email   : zhangchao5@genomics.cn
import math
import torch
import torch.nn as nn


def linear_beta_scheduler(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            timesteps,
            beta_schedule='cosine',
            sampling_timesteps=100,
            objective='pred_noise'
    ):
        super().__init__()
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.model = model
        self.objective = objective

        # define the betas generator
        if beta_schedule == 'linear':
            betas = linear_beta_scheduler(self.timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.timesteps)
        else:
            raise ValueError(f'UNK: {beta_schedule}')

        # define the generate anything noise parameters
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_mius_alphas_bar', torch.sqrt(1. - alphas_bar))

        # define the generate anything denoise parameters
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer('sqrt_recipm1_alpha_bar', torch.sqrt(1. / alphas_bar - 1))

    def diffusion_process(self, x_start, t, noise):
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_bar, t, x_start.shape) * noise
        )
        return x_t

    def forward(self, embed_x1, embed_x2, **kwargs):
        # generate a batch timesteps
        t = torch.randint(0, self.timesteps, (embed_x2.shape[0],), device=embed_x2.device)
        noise = torch.randn_like(embed_x2)
        x_t = self.diffusion_process(x_start=embed_x2, t=t, noise=noise)
        model_output = self.model(x_input=embed_x1, x_target=x_t, t=t, **kwargs)
        if self.objective == 'pred_noise':
            target = x_t
        elif self.objective == 'pred_x0':
            target = embed_x1
        else:
            raise ValueError(f'UNK: {self.objective}')

    # denoise progress
    def ddim_sample(self, embed_x1, **kwargs):
        raise NotImplementedError

    def model_prediction(self, **kwargs):
        raise NotImplementedError

    def pred_start_from_noise(self, x_t, t, pred_noise):
        return (
                extract(self.sqrt_recip_alpha_bar, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * pred_noise
        )

    def pred_noise_from_start(self, x_t, t, x_0):
        return (
                (extract(self.sqrt_recip_alpha_bar, t, x_t.shape) * x_t - x_0) /
                extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape)
        )



