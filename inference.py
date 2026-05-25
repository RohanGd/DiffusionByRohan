from model import DDPM, SinusoidalTimeEmbeddings, UniformNoiseSchedule, CNNNoiseEstimator
from dataload.SmileyDataset import SmileyDataset

import torch
from pathlib import Path

T = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

smiley_dataset = SmileyDataset(len=512, dim=(32, 32))
B, C, H, W = 1, *smiley_dataset.shape

print("Smiley Dataset", smiley_dataset.shape)
noise_schedule = UniformNoiseSchedule(T=T)
time_embedder = SinusoidalTimeEmbeddings(C=64)
noise_estimator = CNNNoiseEstimator(time_embedder=time_embedder, dim=smiley_dataset.shape)

noise_schedule = noise_schedule.to(device)
time_embedder = time_embedder.to(device)
noise_estimator = noise_estimator.to(device)
model = DDPM(T=T, dim=smiley_dataset.shape, noise_schedule=noise_schedule, time_embedder=time_embedder, noise_estimator=noise_estimator)
model = model.to(device)

checkpoint_path = Path("checkpoints/smiley_ddpm.pt")
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}. Run train.py first.")

model.load_state_dict(torch.load(checkpoint_path, map_location=device))

model.eval()

with torch.no_grad():
    xt = model.sample((B, C, H, W), device=device)

    import matplotlib.pyplot as plt 
    xt = xt.squeeze(0)
    img = (xt.permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
    img = img.clip(0, 1)
    plt.imshow(img)
    plt.title(f"Smiley Image - Shape: ")
    plt.axis('off')
    plt.show()
