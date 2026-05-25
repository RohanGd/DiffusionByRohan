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

    @torch.no_grad()
    def sample(self, shape: tuple[int, int, int, int], device=None) -> Tensor:
        device = device or next(self.parameters()).device
        x_t = torch.randn(shape, device=device)
        batch_size = shape[0]

        for step in range(self.T, 0, -1):
            t = torch.full((batch_size, 1), step, device=device, dtype=torch.long)

            alpha_t = self.noise_schedule(t).view(batch_size, 1, 1, 1)
            alpha_bar_t = self.noise_schedule.bar(t).view(batch_size, 1, 1, 1)
            beta_t = 1.0 - alpha_t

            noise_estimate = self(x_t, t)
            mean = (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_estimate) / torch.sqrt(alpha_t)

            if step == 1:
                x_t = mean
                continue

            t_prev = torch.full((batch_size, 1), step - 1, device=device, dtype=torch.long)
            alpha_bar_prev = self.noise_schedule.bar(t_prev).view(batch_size, 1, 1, 1)
            posterior_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            x_t = mean + torch.sqrt(posterior_variance) * torch.randn_like(x_t)

        return x_t
