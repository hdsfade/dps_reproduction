import torch
import torch.nn.functional as F
from .posterior_calculator import PosteriorCalculatorFactory
from .helpers import extract
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import numpy as np
from .helpers import clear_color

class BetaScheduleFactory:
    beta_start = 0.0001
    beta_end = 0.02

    @staticmethod
    def create_schedule(schedule_type, timesteps):
        if schedule_type == 'cosine':
            return BetaScheduleFactory.cosine_beta_schedule(timesteps)
        elif schedule_type == 'linear':
            return BetaScheduleFactory.linear_beta_schedule(timesteps)
        elif schedule_type == 'quadratic':
            return BetaScheduleFactory.quadratic_beta_schedule(timesteps)
        elif schedule_type == 'sigmoid':
            return BetaScheduleFactory.sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def linear_beta_schedule(timesteps):
        return torch.linspace(BetaScheduleFactory.beta_start, BetaScheduleFactory.beta_end, timesteps)

    @staticmethod
    def quadratic_beta_schedule(timesteps):
        return torch.linspace(BetaScheduleFactory.beta_start ** 0.5, 
            BetaScheduleFactory.beta_end ** 0.5, timesteps) ** 2

    @staticmethod
    def sigmoid_beta_schedule(timesteps):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (BetaScheduleFactory.beta_end - BetaScheduleFactory.beta_start) + beta_start

class GaussianDiffusionSampler:
    def __init__(self, betas, mean_calculator, var_calculator):
        self.timesteps = len(betas)
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        self.mean_calculator = mean_calculator
        self.var_calculator = var_calculator

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, model, x, t):
        pass
    
    def p_sample_loop(self, model, x0, y, measurement_cond_fn, record_step, save_root):
        img = x0
        device = x0.device
        timesteps = self.timesteps
        batch_size = x0.shape[0]

        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            t = torch.tensor([i] * batch_size, device=device)
            img = img.requires_grad_()
            out = self.p_sample(model=model, x=img, t=t)
            
            y_n = self.q_sample(y, t=t)
            
            img = measurement_cond_fn(x_t=out['sample'],
                                      y=y,
                                      y_n=y_n,
                                      x_i=img,
                                      x_0_hat=out['pred_x0'])
            img = img.detach_()

            if record_step is not None and record_step != 0 and i % record_step == 0:
                file_path = os.path.join(save_root, f"progress/sample_{i}.png")
                plt.imsave(file_path, clear_color(img))
        return img

    def p_mean_variance(self, model, x, t):
        model_output = model(x, t)
        
        model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        
        mean, pred_x0 = self.mean_calculator.get_mean_and_x0(x=x, t=t, info=model_output)
        var = self.var_calculator.get_variance(model_var_values, t)
        
        return {
            'mean': mean,
            'variance': var,
            'pred_x0': pred_x0
        }
    
class DDPM(GaussianDiffusionSampler):
    def p_sample(self, model, x, t):
        out = self.p_mean_variance(model, x, t)
        sample = out['mean']
        
        if t[0] != 0:
            noise = torch.randn_like(x)
            sample += torch.sqrt(out['variance']) * noise
        
        return {'sample': sample, 'pred_x0': out['pred_x0']}

class SamplerFactory:
    def __init__(self, beta_schedule_factory: BetaScheduleFactory, posterior_calculator_factory: PosteriorCalculatorFactory):
        self.beta_schedule_factory = beta_schedule_factory
        self.posterior_calculator_factory = posterior_calculator_factory
    
    def create_sampler(self, sampler_type, betas_type, noise_steps, mean_type, var_type):
        betas = self.beta_schedule_factory.create_schedule(betas_type, noise_steps)
        mean_calculator = self.posterior_calculator_factory.create_mean_calculator(mean_type, betas)
        var_calculator = self.posterior_calculator_factory.create_var_calculator(var_type, betas)
        if sampler_type == 'ddpm':
            return DDPM(betas, mean_calculator, var_calculator)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")