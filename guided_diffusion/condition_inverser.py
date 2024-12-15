from abc import ABC, abstractmethod
import torch


class ConditionInverser(ABC):
    def __init__(self, operator, noiser, scale=1.0):
        self.operator = operator
        self.noiser = noiser
        self.scale = scale
        
    def project(self, x, y_n):
        return self.operator.project(x=x, y=y_n)
    
    def distance(self, x_0_hat, y):
        return torch.linalg.norm(y - self.operator.forward(x_0_hat))
    
    def grad(self,  x_i, x_0_hat, y):
        outputs = torch.linalg.norm(y - self.operator.forward( x_0_hat))
        
        if self.noiser.__name__ == 'poisson':
            outputs = (outputs / y.abs()).mean()
        grad = torch.autograd.grad(outputs, x_i)[0]
        
        return grad
    
    @abstractmethod
    def condition_inverse(self, x_i, x_0_hat, y, y_n=None):
        pass
    
class MCGConditionInverser(ConditionInverser):
    def __init__(self, operator, noiser, scale=1.0):
        super().__init__(operator, noiser, scale)
        
    def condition_inverse(self, x_i, x_t, x_0_hat, y, y_n):
        grad = self.grad(x_i, x_0_hat, y)
        x_t -= self.scale * grad
        x_t = self.project(x_t, y_n)
        return x_t

class PosteriorSampingConditionInverser(ConditionInverser):
    def __init__(self, operator, noiser, scale=1.0):
        super().__init__(operator, noiser, scale)
        
    def condition_inverse(self, x_i, x_t, x_0_hat, y, y_n):
        grad = self.grad(x_i, x_0_hat, y)
        x_t = x_t - self.scale * grad
        return x_t

class ConditionInverserFactory:
    @staticmethod
    def create_condition_inverser(name, operator, noiser, scale=1.0):
        if name == 'mcg':
            return MCGConditionInverser(operator, noiser, scale)
        elif name == 'ps':
            return PosteriorSampingConditionInverser(operator, noiser, scale)
        else:
            raise ValueError(f"Unknown ConditionInverse Type: {name}")