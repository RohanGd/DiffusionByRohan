from model import DDPM, SinusoidalTimeEmbeddings, UniformNoiseSchedule, CNNNoiseEstimator
from dataload.forwardDiffusionProcess import forward_diffusion_process
from dataload.SmileyDataset import SmileyDataset

from torch.utils.data import DataLoader
import torch
from pathlib import Path

num_epochs = 10
T = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

smiley_dataset = SmileyDataset(len=10000, dim=(32, 32))
dataloader = DataLoader(dataset=smiley_dataset, batch_size=4, shuffle=True)

print("Smiley Dataset", smiley_dataset.shape)
noise_schedule = UniformNoiseSchedule(T=T)
time_embedder = SinusoidalTimeEmbeddings(C=256)
noise_estimator = CNNNoiseEstimator(time_embedder=time_embedder, dim=smiley_dataset.shape)

noise_schedule = noise_schedule.to(device)
time_embedder = time_embedder.to(device)
noise_estimator = noise_estimator.to(device)
model = DDPM(T=T, dim=smiley_dataset.shape, noise_schedule=noise_schedule, time_embedder=time_embedder, noise_estimator=noise_estimator)
model = model.to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=2e-4)
checkpoint_path = Path("checkpoints/smiley_ddpm.pt")
sample_dir = Path("samples")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
sample_dir.mkdir(parents=True, exist_ok=True)

# print(model)
print(sum([torch.numel(m) for m in model.parameters()]))
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for idx, batch in enumerate(dataloader):
        B, C, H, W = batch.shape
        optimizer.zero_grad()

        x0 = batch.to(device)
        t = torch.randint(low=1, high=T + 1, size=(B, 1), dtype=torch.long, device=x0.device)
        xt, epsilon = forward_diffusion_process(noise_scheduler=noise_schedule, x0=x0, t=t)

        noise_estimate = model(xt, t)
        
        loss = torch.nn.functional.mse_loss(noise_estimate, epsilon)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"epoch {epoch:04d} loss {epoch_loss / len(dataloader):.6f}")

    # if epoch % 1 != 0:
    #     continue

    torch.save(model.state_dict(), checkpoint_path)
    model.eval()
    with torch.no_grad():
        xt = model.sample((1, C, H, W), device=device)

        import matplotlib.pyplot as plt 
        xt = xt.squeeze(0)
        img = (xt.permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
        img = img.clip(0, 1)
        plt.imshow(img)
        plt.title(f"Smiley Image - Epoch: {epoch}")
        plt.axis('off')
        plt.savefig(sample_dir / f"epoch_{epoch:04d}.png")
        plt.close()
        
