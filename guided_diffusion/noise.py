import numpy as np
import torch
from abc import ABC, abstractmethod

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

class Clean(Noise):
    def __init__(self):
        self.__name__ = 'clean'
    def forward(self, data):
        return data

class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
        self.__name__ = 'gaussian'
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate
        self.__name__ = 'poisson'

    def forward(self, data):

        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

class NoiseFactory:
    @staticmethod
    def create_noise(name, **kwargs):
        if name == 'clean':
            return Clean()
        elif name == 'gaussian':
            return GaussianNoise(**kwargs)
        elif name == 'poisson':
            return PoissonNoise(**kwargs)
        else:
            raise ValueError(f"Unknown noise type: {name}")