import torch
import torch.nn as nn
from torch import Tensor
from model.noise_estimator import CNNNoiseEstimator

class DDPM(nn.Module):
    def __init__(self, T:int, dim:tuple, noise_schedule, time_embedder, noise_estimator):
        """
        Args: T(int) number of time steps

        """
        super().__init__()
        self.T = T
        self.noise_schedule = noise_schedule
        self.noise_estimator = noise_estimator
        self.time_embedder = time_embedder

    def forward(self, x_t, t):

        noise_estimate = self.noise_estimator(x_t, t)
        return noise_estimate

