from torch.utils.data import Dataset
from torchvision.io import read_image
import torch.nn.functional as F
import torch


class SmileyDataset(Dataset):
    def __init__(self, len=512, dim=(32, 32)):
        super().__init__()
        self.len = len
        img_path = "data/smiley.jpg"
        self.smiley = read_image(img_path).to(dtype=torch.float32)
        self.smiley = F.interpolate(self.smiley.unsqueeze(0), size=dim, mode='bilinear', align_corners=False).squeeze(0)
        self.smiley_normalized = (self.smiley / 127.5) - 1.0
        self.shape = self.smiley.shape

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.smiley_normalized

    def show(self):
        import matplotlib.pyplot as plt 
        img = self.smiley.permute(1, 2, 0) / 255.0
        plt.imshow(img)
        plt.title(f"Smiley Image - Shape: {self.smiley.shape}")
        plt.axis('off')
        plt.show()




if __name__ == "__main__":
    smiley_d = SmileyDataset()
    smiley_d.show()