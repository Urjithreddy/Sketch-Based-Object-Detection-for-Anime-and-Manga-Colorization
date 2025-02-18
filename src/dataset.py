import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ColorizationDataset(Dataset):
    def __init__(self, dataset, img_size):
        self.dataset = dataset
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        bw_image = Image.fromarray(np.array(sample['bw_image'])).convert("RGB")
        color_image = Image.fromarray(np.array(sample['color_image'])).convert("RGB")

        bw_image = bw_image.resize((self.img_size, self.img_size))
        color_image = color_image.resize((self.img_size, self.img_size))

        bw_image = np.array(bw_image).transpose((2, 0, 1)) / 255.0
        color_image = np.array(color_image).transpose((2, 0, 1)) / 255.0

        bw_image = torch.tensor(bw_image, dtype=torch.float32)
        color_image = torch.tensor(color_image, dtype=torch.float32)

        return bw_image, color_image