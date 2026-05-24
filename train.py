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
optimizer = torch.optim.AdamW(params=model.parameters())


for epoch in range(num_epochs):
    model.train()
    for idx, batch in enumerate(dataloader):
        # should the dataset have already the x_0 and the corresponding epsilon, t and xt?
        # or should it be processed inside this loop?
        # if it is in the dataset, then the noise is constant, but here it will be different as it will be sampled each time from a gaussian.

        B, C, H, W = batch.shape
        optimizer.zero_grad()
        # 1. sampling clean images
        x0 = batch.to(device)
        # 1. sampling a noise from a gaussian
        epsilon = torch.randn_like(x0)
        # 1. sampling a time step
        t = torch.randint(low=1, high=T, size=(B, 1)).to(dtype=torch.long, device=x0.device)
        # t = torch.arange(B, dtype=batch.dtype).reshape((B,1))
        # t = t.to(device)

        # 1. create the noise image:
        xt, epsilon = forward_diffusion_process(noise_scheduler=noise_schedule, x0=x0, t=t)
        # alpha_bar_t = noise_schedule.bar(t).view((B, 1, 1, 1))

        # xt =  torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1-alpha_bar_t) * epsilon
        
        noise_estimate = model(xt, t)
        
        loss = torch.nn.functional.mse_loss(noise_estimate, epsilon)
        print(loss.item())
        loss.backward()
        optimizer.step()

    if epoch % 50 != 0:
        continue

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
        