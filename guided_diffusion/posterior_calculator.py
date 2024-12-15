from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from .helpers import extract

class PosteriorMeanCalculator(ABC):
    def __init__(self, betas):
        """
        Initialize the PosteriorMeanCalculator with given betas.
        
        Args:
            betas (torch.Tensor): Tensor of betas.
        """
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    @abstractmethod
    def get_mean_and_x0(self, x, t, info=None):
        """
        Get the mean and x0.
        
        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Time step tensor.
            eps (torch.Tensor): Noise tensor.
        
        Returns:
            tuple: Mean and predicted x0 tensors.
        """
        pass
    
class PosteriorMeanEpsCalculator(PosteriorMeanCalculator):
    def __init__(self, betas):
        """
        Initialize the PosteriorMeanEpsCalculator with given betas.
        
        Args:
            betas (torch.Tensor): Tensor of betas.
        """
        super().__init__(betas)
        self.q_posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.q_posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        self.predict_x0_coef1 = 1.0 / self.sqrt_alphas_cumprod
        self.predict_x0_coef2 = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod
    
    def predict_x0(self, x_t, t, eps):
        """
        Predict x0 from x_t and eps.
        
        Args:
            x_t (torch.Tensor): Tensor at time t.
            t (torch.Tensor): Time step tensor.
            eps (torch.Tensor): Noise tensor.
        
        Returns:
            torch.Tensor: Predicted x0 tensor.
        """
        coef1 = extract(self.predict_x0_coef1, t, x_t.shape)
        coef2 = extract(self.predict_x0_coef2, t, x_t.shape)
        x0 = coef1 * x_t - coef2 * eps
        return x0.clamp(-1, 1)
        
    def q_posterior_mean(self, x_0, x_t, t):
        """
        Calculate the posterior mean: q(x_{t-1} | x_t, x_0).
        
        Args:
            x_0 (torch.Tensor): Tensor at time 0.
            x_t (torch.Tensor): Tensor at time t.
            t (torch.Tensor): Time step tensor.
        
        Returns:
            torch.Tensor: Posterior mean tensor.
        """
        coef1 = extract(self.q_posterior_mean_coef1, t, x_0.shape)
        coef2 = extract(self.q_posterior_mean_coef2, t, x_t.shape)
        return coef1 * x_0 + coef2 * x_t
    
    def get_mean_and_x0(self, x, t, info):
        """
        Get the mean and x0.
        
        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Time step tensor.
            eps (torch.Tensor): Noise tensor.
        
        Returns:
            tuple: Mean and predicted x0 tensors.
        """
        eps = info
        pred_x0 = self.predict_x0(x, t, eps)
        mean = self.q_posterior_mean(pred_x0, x, t)
        return mean, pred_x0
    
class PosteriorVarCalculator(ABC):
    def __init__(self, betas):
        """
        Initialize the PosteriorVarCalculator with given betas.
        
        Args:
            betas (torch.Tensor): Tensor of betas.
        """
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
    @abstractmethod
    def get_variance(self, var_info, t):
        """
        Get the variance and log variance.
        
        Args:
            var_prop (torch.Tensor): Variance proportion tensor.
            t (torch.Tensor): Time step tensor.
        
        Returns:
            tuple: Variance and log variance tensors.
        """
        pass

class PosteriorVarPropCalculator(PosteriorVarCalculator):
    def __init__(self, betas):
        """
        Initialize the PosteriorVarPropCalculator with given betas.
        
        Args:
            betas (torch.Tensor): Tensor of betas.
        """
        super().__init__(betas)
        q_posterior_var = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.q_posterior_log_var_clipped = torch.log(torch.cat([q_posterior_var[1:2], q_posterior_var[1:]]))
        
    def q_posterior_var(self, var_info, t):
        """
        Calculate the posterior variance: q(x_{t-1} | x_t, x_0).
        
        Args:
            var_info (torch.Tensor): Variance proportion tensor.
            t (torch.Tensor): Time step tensor.
        
        Returns:
            tuple: Variance and log variance tensors.
        """
        var_prop = var_info
        min_log_var = extract(self.q_posterior_log_var_clipped, t, var_prop.shape)
        max_log_var = extract(torch.log(self.betas), t, var_prop.shape)
        
        frac = (var_prop + 1.0) / 2.0
        log_var = frac * max_log_var + (1.0 - frac) * min_log_var
        var = torch.exp(log_var)
        return var
    
    def get_variance(self, var_prop, t):
        """
        Get the variance and log variance.
        
        Args:
            var_prop (torch.Tensor): Variance proportion tensor.
            t (torch.Tensor): Time step tensor.
        
        Returns:
            tuple: Variance and log variance tensors.
        """
        return self.q_posterior_var(var_prop, t)

class PosteriorCalculator:
    def __init__(self, mean_calculator: PosteriorMeanCalculator, var_calculator: PosteriorVarCalculator):
        """
        Initialize the PosteriorCalculator with given mean and variance calculators.
        
        Args:
            mean_calculator (PosteriorMeanCalculator): Mean calculator.
            var_calculator (PosteriorVarCalculator): Variance calculator.
        """
        self.mean_calculator = mean_calculator
        self.var_calculator = var_calculator
          
    def get_mean_and_x0(self, x, t, eps):
        return self.mean_calculator.get_mean_and_x0(x, t, eps)

    def get_variance(self, var_prop, t):
        return self.var_calculator.get_variance(var_prop, t)

class PosteriorCalculatorFactory:
    @staticmethod
    def create_mean_calculator(calculator_type, betas):
        if calculator_type == 'eps':
            return PosteriorMeanEpsCalculator(betas)
        else:
            raise ValueError(f"Unknown mean calculator type: {calculator_type}")
    
    @staticmethod
    def create_var_calculator(calculator_type, betas):
        if calculator_type == 'prop':
            return PosteriorVarPropCalculator(betas)
        else:
            raise ValueError(f"Unknown variance calculator type: {calculator_type}")
