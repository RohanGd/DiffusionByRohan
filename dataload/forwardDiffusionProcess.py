from model import DDPMNoiseSchedule
import torch

def forward_diffusion_process(noise_scheduler, x0: torch.Tensor, t):
    '''
    Args:
        x_0 = [C, H, W]
    Returns:
        x1: [C, H, W], device = x_0.device
        noise: [C, H, W], device = x_0.device
    '''
    B, C, H, W = x0.shape
    alpha_bar_t = noise_scheduler.bar(t)
    coeff_x0 = torch.sqrt(alpha_bar_t).view(B, 1, 1, 1) # sqroot(alphabar_t)

    noise = torch.randn_like(x0)
    noise_coeff = torch.sqrt(1 - alpha_bar_t).view(B, 1, 1, 1)

    x1 = coeff_x0 * x0 + noise_coeff * noise
    return x1, noise
