from abc import ABC, abstractmethod
from functools import partial
import kornia
import torch
import torch.nn.functional as F

class LinearOperator(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def transpose(self, x):
        pass

    def ortho_project(self, x):
        return x - self.transpose(self.forward(x))

    def project(self, x, y): 
        return  self.ortho_project(y) - self.forward(x)

class InpaintingOperator(LinearOperator):
    def __init__(self, device):
        self.device = device

    def set_mask(self, mask):
        self.mask = mask
        
    def forward(self, x):
        return x * self.mask.to(self.device)
    
    def transpose(self, x):
        return x

    def project(self, x, y):
        return x - self.transpose(self.forward(x)) + self.transpose(y)

class SuperResolutionOperator(LinearOperator):
    def __init__(self, scale_factor, device):
        self.device = device
        self.down_sample = partial(F.interpolate, scale_factor=1/scale_factor)
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        
    def forward(self, x):
        return self.down_sample(x)
    
    def transpose(self, x):
        return self.up_sample(x)

    def project(self, x, y):
        return x - self.transpose(self.forward(x)) + self.transpose(y)

class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, direction, device):
        self.device = device
        self.kernel_size = kernel_size
    
        self.conv = partial(kornia.filters.motion_blur, kernel_size=kernel_size, angle=0, direction=direction)

    def forward(self, x):
        return  self.conv(x)

    def transpose(self, x):
        return x

class GaussianBlurOperator(LinearOperator):
    def __init__(self, kernel_size, sigma, device):
        self.device = device
        self.kernel_size = kernel_size
    
        self.conv = partial(kornia.filters.gaussian_blur2d, kernel_size=kernel_size, sigma=(sigma, sigma))

    def forward(self, x):
        return  self.conv(x)

    def transpose(self, x):
        return x

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def project(self, x, y):
        return x + y - self.forward(x) 

class MosaicOperator(LinearOperator):
    def __init__(self, mosaic_size, device):
        self.mosaic_size = mosaic_size
        self.device = device

    def forward(self, x):
        N, C, H, W = x.shape
        mosaic_size = self.mosaic_size

        mosaic_x = x.unfold(2, mosaic_size, mosaic_size).unfold(3, mosaic_size, mosaic_size)
        mosaic_x = mosaic_x.contiguous().view(N, C, -1, mosaic_size, mosaic_size)
        mosaic_x = mosaic_x.mean(dim=(3, 4), keepdim=True)
        mosaic_x = mosaic_x.repeat(1, 1, 1, mosaic_size, mosaic_size)

        mosaic_x = mosaic_x.view(N, C, H // mosaic_size, W // mosaic_size, mosaic_size, mosaic_size)
        mosaic_x = mosaic_x.permute(0, 1, 2, 4, 3, 5).contiguous().view(N, C, H, W)

        return mosaic_x

    def transpose(self, x):
        return x
class GammaBlurOperator(NonLinearOperator):
    def __init__(self, gamma, gain, device):
        self.conv = partial(kornia.enhance.adjust_gamma, gamma=gamma, gain=gain)
        self.device = device

    def forward(self, x):
        x = (x + 1) / 2
        x = self.conv(x)
        return x * 2 - 1

class GrayOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, x):
        grayscale_x = x[:, 0, :, :] * 0.2989 + x[:, 1, :, :] * 0.5870 + x[:, 2, :, :] * 0.1140
        grayscale_x = grayscale_x.unsqueeze(1)
        grayscale_x = grayscale_x.repeat(1, x.shape[1], 1, 1)
        return grayscale_x
    def transpose(self, x):
        return x

class ColorReduceOperator(NonLinearOperator):
    def __init__(self, factor, device):
        self.factor = factor
        self.device = device
    def forward(self, x):
        reduced_x = (x + 1) / 2
        reduced_x = (reduced_x * self.factor).round() / self.factor
        reduced_x = reduced_x * 2 - 1
        return reduced_x



class OperatorFactory:
    @staticmethod
    def create_operator(name, **kwargs):
        if name == 'inpainting':
            return InpaintingOperator(**kwargs)
        elif name == 'super_resolution':
            return SuperResolutionOperator(**kwargs)
        elif name == 'motion_blur':
            return MotionBlurOperator(**kwargs)
        elif name == 'gaussian_blur':
            return GaussianBlurOperator(**kwargs)
        elif name == 'gamma_blur':
            return GammaBlurOperator(**kwargs)
        elif name == 'mosaic':
            return MosaicOperator(**kwargs)
        elif name == 'gray':
            return GrayOperator(**kwargs)
        elif name == 'color_reduce':
            return ColorReduceOperator(**kwargs)
        else:
            raise ValueError(f"Unknown operator type: {name}")
