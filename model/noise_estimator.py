import torch
from torch import nn, Tensor
from model.time_embeddings import TimeEmbeddings


class CNNNoiseEstimator(nn.Module):
    def __init__(self, time_embedder: TimeEmbeddings, dim: tuple, hidden_ch: int = 64, num_blocks: int = 6):
        super().__init__()
        self.C, self.H, self.W = dim
        self.time_embedder = time_embedder

        self.stem = nn.Conv2d(self.C, hidden_ch, kernel_size=3, padding=1)

        self.convs      = nn.ModuleList([nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1) for _ in range(num_blocks)])
        self.time_projs = nn.ModuleList([nn.Linear(time_embedder.C, hidden_ch) for _ in range(num_blocks)])

        self.out = nn.Conv2d(hidden_ch, self.C, kernel_size=1) # no activation, output must be unbounded

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_embedder(t) # (B, time_emb_dim)

        x = torch.sigmoid(self.stem(x_t)) # (B, hidden_ch, H, W)
        # res = torch.clone(x).detach() # add residuals
        # b = True
        for conv, time_proj in zip(self.convs, self.time_projs):
            x = x + time_proj(t_emb)[:, :, None, None] # inject time at every layer
            x = torch.tanh(conv(x))
            # if b:
            #     x = x + res 
            #     b = False
            # if not b:
            #     b = True

        return self.out(x) # (B, C, H, W) — linear, no activation