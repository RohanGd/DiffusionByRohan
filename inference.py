from model import DDPM, SinusoidalTimeEmbeddings, UniformNoiseSchedule, CNNNoiseEstimator
from dataload.forwardDiffusionProcess import forward_diffusion_process
from dataload.SmileyDataset import SmileyDataset

from torch.utils.data import DataLoader
import torch

num_epochs = 500
T = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

smiley_dataset = SmileyDataset(len=512, dim=(32, 32))
dataloader = DataLoader(dataset=smiley_dataset, batch_size=4)

print("Smiley Dataset", smiley_dataset.shape)
noise_schedule = UniformNoiseSchedule(T=T)
time_embedder = SinusoidalTimeEmbeddings(C=64)
noise_estimator = CNNNoiseEstimator(time_embedder=time_embedder, dim=smiley_dataset.shape)

noise_schedule = noise_schedule.to(device)
time_embedder = time_embedder.to(device)
noise_estimator = noise_estimator.to(device)
model = DDPM(T=T, dim=smiley_dataset.shape, noise_schedule=noise_schedule, time_embedder=time_embedder, noise_estimator=noise_estimator)
model = model.to(device)


model.eval()

with torch.no_grad():
    xt = torch.randn((1, C, H, W), device=device)

    for t in range(T - 1, -1, -1):
        alpha_t = noise_schedule(t)
        alpha_bar_t = noise_schedule.bar(t)

        coeff_1 = 1 / torch.sqrt(alpha_t)
        coeff_2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

        t_tensor      = torch.tensor([[t]], device=device)
        noise_estimate = model(xt, t_tensor)
        xt = coeff_1 * (xt - coeff_2 * noise_estimate)

        if t > 0:  # no noise added at the final step
            sigma_t = torch.sqrt(1 - alpha_t)
            xt = xt + sigma_t * torch.randn_like(xt)

    import matplotlib.pyplot as plt 
    xt = xt.squeeze(0)
    img = (xt.permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
    img = img.clip(0, 1)
    plt.imshow(img)
    plt.title(f"Smiley Image - Shape: ")
    plt.axis('off')
    plt.show()