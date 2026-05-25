from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class DDPMNoiseSchedule(nn.Module, ABC):
    def __init__(self):
        super().__init__() # Initializes nn.Module internals
        
    @abstractmethod
    def bar(self, t):
        """Returns alpha_bar for given timestep t"""
        pass
    
    @abstractmethod
    def forward(self, t):
        """Returns alpha for given timestep t"""
        pass

class UniformNoiseSchedule(DDPMNoiseSchedule):
    def __init__(self, T: int, beta_val: float = 0.02):
        super().__init__() # Calls DDPMNoiseSchedule's init
        self.T = T
        
        # True uniform schedule uses a constant value across all steps
        betas = torch.full((T,), beta_val, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        # register_buffer ensures these tensors move to GPU automatically
        # when you call `.to(device)` on your parent model.
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        
    def forward(self, t: torch.Tensor):
        """ Expects t in range [1, T]"""
        has_negative = (t-1 < 0).any().item()
        assert has_negative != True, "negative index t. t expects to be in range [1, T] not 0"
        return self.alphas[t-1]
    
    def bar(self, t: torch.Tensor):
        """ Expects t in range [1, T]"""
        has_negative = (t-1 < 0).any().item()
        assert has_negative != True, "negative index t. t expects to be in range [1, T] not 0"
        return self.alpha_bars[t-1]
