import torch
from torch import Tensor, nn
from abc import ABC


class TimeEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()


class SinusoidalTimeEmbeddings(TimeEmbeddings):
    """
    Args: __init__: C:int (channels)
    Returns: (B, C) per time dim a int for each channel.
    Given a time step t, map it to a high dimensional vector of size D using sinusoidal signals of varying frequencies.
    t -> vector_D ---MLP---> vector_C
    C is the hidden channels of your noise_estimator
    So per channel we broadcast the time embedding across all pixels, thereby encoding a time embedding for each channel.
    """

    def __init__(self, C: int = 64):
        super().__init__()
        self.D = 256
        self.C = C

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.D, out_features=self.C, bias=False), nn.Sigmoid()
        )

    def forward(self, t: Tensor):
        """
        Args: t (B, 1)
        Returns: (B, C)
        """
        if t.ndim != 2:
            t = t.unsqueeze(dim=1)
            assert t.ndim == 2, "t must have shape (B, 1)"

        # pe(t)[2i] = sin(t / 10000 ^ 2i/D), pe(t)[2i+1] = cos(t / 10000 ^ 2i/D)
        B, _ = t.shape
        time_emb = torch.zeros((B, self.D), device=t.device, dtype=torch.float32)
        i = torch.arange(
            self.D // 2, 
            dtype=torch.float32,
            device=t.device
        )  # i is a vector of size D/2. since each i will generate one "sine" and one "cos" emb it is D/2
        # for each i generate sin and cos for the angle: t / 10000 ** 2i/D
        denominator = (
            10000 ** (2 * i / self.D)
        )  # torch.tensor handles broadcast. i is a vector of size D7", denom will also be a vector of size D/2
        angles = t / denominator

        # every alternate time step
        time_emb[:, 0::2] = torch.sin(angles)
        time_emb[:, 1::2] = torch.cos(angles)

        # forward pass through the mlp. evrything before is not part of gradient update
        time_emb = self.mlp(time_emb)
        return time_emb
