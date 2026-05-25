import torch
from torch import nn, Tensor
from model.time_embeddings import TimeEmbeddings


class CNNNoiseEstimator(nn.Module):
    def __init__(self, time_embedder: TimeEmbeddings, dim: tuple, hidden_ch: int = 256, num_blocks: int = 8):
        super().__init__()
        self.C, self.H, self.W = dim
        self.time_embedder = time_embedder

        self.stem = nn.Conv2d(self.C, hidden_ch, kernel_size=3, padding=1)

        self.convs = nn.ModuleList([nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1) for _ in range(num_blocks)])
        self.time_projs = nn.ModuleList([nn.Linear(time_embedder.C, hidden_ch) for _ in range(num_blocks)])
        self.out = nn.Conv2d(hidden_ch, self.C, kernel_size=1) # no activation, output must be unbounded

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_embedder(t) # (B, time_emb_dim)

        x = torch.sigmoid(self.stem(x_t)) # (B, hidden_ch, H, W)
        for conv, time_proj in zip(self.convs, self.time_projs):
            residue = x
            x = x + time_proj(t_emb)[:, :, None, None] # inject time at every layer
            x = torch.tanh(conv(x))
            x = x + residue

        return self.out(x) # (B, C, H, W) — linear, no activation


class CNNNoiseEstimator_v2(nn.Module):
    def __init__(self, time_embedder: TimeEmbeddings, dim: tuple, hidden_ch: int = 256, num_blocks: int = 8):
        super().__init__()
        self.C, self.H, self.W = dim
        self.time_embedder = time_embedder

        self.stem      = nn.Conv2d(self.C, hidden_ch, kernel_size=3, padding=1)
        self.stem_norm = nn.GroupNorm(8, hidden_ch)

        self.convs      = nn.ModuleList([nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1) for _ in range(num_blocks)])
        self.norms      = nn.ModuleList([nn.GroupNorm(8, hidden_ch) for _ in range(num_blocks)])
        self.time_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(time_embedder.C, hidden_ch), nn.SiLU())
            for _ in range(num_blocks)
        ])

        self.out = nn.Conv2d(hidden_ch, self.C, kernel_size=1)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_embedder(t)                           # (B, time_emb_dim)
        x = torch.nn.functional.silu(self.stem_norm(self.stem(x_t)))             # (B, hidden_ch, H, W)

        for conv, norm, time_proj in zip(self.convs, self.norms, self.time_projs):
            residue = x
            x = x + time_proj(t_emb)[:, :, None, None]        # time injection
            x = torch.nn.functional.silu(norm(conv(x)))                          # conv -> norm -> activate
            x = x + residue                                    # residual

        return self.out(x)                                     # (B, C, H, W), linear



## The UNET is borrowed from a AI generated code:
# Building block: ResBlock with time conditioning

class ResBlock(nn.Module):
    """
    A single residual block that:
      1. Applies two 3×3 padded convolutions (spatial dims are preserved).
      2. Injects the time embedding between the two convolutions via a
         learned linear projection, broadcast across H×W.
      3. Adds a 1×1 identity/projection shortcut so in_ch need not equal out_ch.

    Spatial dims are always preserved (padding=1 on every 3×3 conv).
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()

        # ── first half ──────────────────────────────────────────────────────
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # time projection: maps (B, time_emb_dim) -> (B, out_ch)
        # result is added as (B, out_ch, 1, 1) -> broadcasts over H×W
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        # ── second half ─────────────────────────────────────────────────────
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # ── shortcut (identity if channels match, 1×1 conv otherwise) ───────
        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        self.act = nn.SiLU()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x     : (B, in_ch,  H, W)
            t_emb : (B, time_emb_dim)
        Returns:
            out   : (B, out_ch, H, W)
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # inject time — shape (B, out_ch) -> (B, out_ch, 1, 1)
        h = h + self.time_proj(t_emb)[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.shortcut(x)


# Down / Up sampling for multi-scale feature learning

class Downsample(nn.Module):
    """Stride-2 conv — halves H and W, preserves channels."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1) # (w - K + 2p)/ s + 1 = (32 - 3 + 2) / 2 + 1 = 16

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x nearest upsample + conv — doubles H and W, preserves channels."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1) # (w - K + 2p)/ s + 1 = (32 - 3 + 2) / 1 + 1 = 32

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest") # upsamples by 2
        return self.conv(x)


# U-Net noise estimator
class UNetNoiseEstimator(nn.Module):
    """
    Lightweight U-Net noise estimator for DDPM.

    Architecture
    stem  ->  [enc_1]  ->  down  ->  [enc_2]  ->  down  ->  [bottleneck]
                                                              |
    out   <-  [dec_1]  <-  up   <-  [dec_2]  <-  up   <-  (skip cats)

    """

    def __init__(
        self,
        time_embedder: TimeEmbeddings,
        dim: tuple, # (C, H, W) of x_t
        base_ch: int = 64, # channels at the first encoder level
    ):
        super().__init__()
        self.C, self.H, self.W = dim
        self.time_embedder = time_embedder
        time_emb_dim: int = time_embedder.C  # must match TimeEmbeddings output dim

        ch1 = base_ch       # 64
        ch2 = base_ch * 2   # 128
        ch3 = base_ch * 4   # 256  (bottleneck)

        # ── stem: map image channels -> ch1, no spatial change ───────────────
        self.stem = nn.Conv2d(self.C, ch1, kernel_size=3, padding=1)

        # ── encoder ─────────────────────────────────────────────────────────
        self.enc1_a = ResBlock(ch1, ch1, time_emb_dim)
        self.enc1_b = ResBlock(ch1, ch1, time_emb_dim)
        self.down1  = Downsample(ch1)                    # H/2

        self.enc2_a = ResBlock(ch1, ch2, time_emb_dim)
        self.enc2_b = ResBlock(ch2, ch2, time_emb_dim)
        self.down2  = Downsample(ch2)                    # H/4

        # ── bottleneck ───────────────────────────────────────────────────────
        self.bot_a = ResBlock(ch2, ch3, time_emb_dim)
        self.bot_b = ResBlock(ch3, ch3, time_emb_dim)
        self.bot_c = ResBlock(ch3, ch2, time_emb_dim)   # project back to ch2

        # ── decoder ─────────────────────────────────────────────────────────
        # After upsample + skip concat the input channel count doubles.
        self.up2    = Upsample(ch2)
        self.dec2_a = ResBlock(ch2 + ch2, ch2, time_emb_dim)  # cat with enc2 skip
        self.dec2_b = ResBlock(ch2,       ch1, time_emb_dim)

        self.up1    = Upsample(ch1)
        self.dec1_a = ResBlock(ch1 + ch1, ch1, time_emb_dim)  # cat with enc1 skip
        self.dec1_b = ResBlock(ch1,       ch1, time_emb_dim)

        # ── output head: linear projection, NO activation ───────────────────
        # ε ~ N(0, 1) is symmetric — an activation would prevent predicting
        # negative noise.
        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=ch1)
        self.out_conv = nn.Conv2d(ch1, self.C, kernel_size=1)



    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x_t : (B, C, H, W)  — noisy image at timestep t
            t   : (B, 1)        — integer timestep indices
        Returns:
            noise  : (B, C, H, W)  — predicted noise (unbounded, no activation)
        """
        # ── time embedding: (B, 1) -> (B, time_emb_dim) ─────────────────────
        t_emb = self.time_embedder(t)   # (B, C_time)

        # ── stem ────────────────────────────────────────────────────────────
        x = self.stem(x_t)              # (B, ch1, H,   W)

        # ── encoder ─────────────────────────────────────────────────────────
        x = self.enc1_a(x, t_emb)      # (B, ch1, H,   W)
        skip1 = self.enc1_b(x, t_emb)  # (B, ch1, H,   W)  <- skip connection
        x = self.down1(skip1)           # (B, ch1, H/2, W/2)

        x = self.enc2_a(x, t_emb)      # (B, ch2, H/2, W/2)
        skip2 = self.enc2_b(x, t_emb)  # (B, ch2, H/2, W/2)  <- skip connection
        x = self.down2(skip2)           # (B, ch2, H/4, W/4)

        # ── bottleneck ───────────────────────────────────────────────────────
        x = self.bot_a(x, t_emb)       # (B, ch3, H/4, W/4)
        x = self.bot_b(x, t_emb)       # (B, ch3, H/4, W/4)
        x = self.bot_c(x, t_emb)       # (B, ch2, H/4, W/4)

        # ── decoder ─────────────────────────────────────────────────────────
        x = self.up2(x)                             # (B, ch2, H/2, W/2)
        x = torch.cat([x, skip2], dim=1)            # (B, ch2+ch2, H/2, W/2)
        x = self.dec2_a(x, t_emb)                  # (B, ch2, H/2, W/2)
        x = self.dec2_b(x, t_emb)                  # (B, ch1, H/2, W/2)

        x = self.up1(x)                             # (B, ch1, H,   W)
        x = torch.cat([x, skip1], dim=1)            # (B, ch1+ch1, H,   W)
        x = self.dec1_a(x, t_emb)                  # (B, ch1, H,   W)
        x = self.dec1_b(x, t_emb)                  # (B, ch1, H,   W)

        # ── output ───────────────────────────────────────────────────────────
        x = torch.nn.functional.silu(self.out_norm(x))
        epsilon_hat = self.out_conv(x)              # (B, C, H, W) — linear!
        return epsilon_hat

