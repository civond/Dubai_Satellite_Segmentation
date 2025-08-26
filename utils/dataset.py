from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
from utils.mask2label import mask2label
import torch

class ImageDataset(Dataset):
    def __init__(self, data_dir, train=True, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(data_dir) if f.endswith(".jpg")])
        self.masks = sorted([f for f in os.listdir(data_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.images[index])
        mask_path = os.path.join(self.data_dir, self.masks[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))

        label = mask2label(mask)
        label = torch.tensor(label, dtype=torch.long)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
     
        return image, label
