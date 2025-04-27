import random
import os
import torch
from torch.utils.data import Dataset, random_split
from typing import Any, Tuple
import torchvision.transforms as transforms
from PIL import Image
import math
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Root directory containing 'normal', 'malignant', 'benign'.
            transform (callable, optional): Transform to apply to images.
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.mask_paths = []

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        self.image_paths = []
        self.mask_paths = []

        # Iterate over normal, malignant, benign folders
        for category in ['normal', 'malignant', 'benign']:
            category_path = os.path.join(root_dir, category)
            images = sorted([f for f in os.listdir(category_path) if "_mask" not in f])  # Exclude masks

            for img in images:
                img_path = os.path.join(category_path, img)
                mask_path = os.path.join(category_path, img.replace('.png', '_mask.png'))  # Assuming PNG format

                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        # Shuffle the dataset before splitting
        combined = list(zip(self.image_paths, self.mask_paths))
        random.shuffle(combined)
        self.image_paths, self.mask_paths = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Convert mask to grayscale

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask


class CustomDataLoader:
    def __init__(self, dataset: Any, batch_size: int = 8, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self._reset()

    def _reset(self) -> None:
        """Prepare indices for a new iteration (epoch)"""
        if self.shuffle:
            random.shuffle(self.indices)
        self.current = 0

    def __iter__(self) -> 'CustomDataLoader':
        self._reset()
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch the next batch of (images, masks) as stacked tensors.
        Raises StopIteration when dataset is exhausted.
        """
        if self.current >= len(self.indices):
            raise StopIteration

        start = self.current
        end = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        self.current = end

        images = []
        masks = []
        for idx in batch_indices:
            img, msk = self.dataset[idx]
            images.append(img)
            masks.append(msk)

        batch_images = torch.stack(images, dim=0)
        batch_masks = torch.stack(masks, dim=0)

        return batch_images, batch_masks

    def __len__(self) -> int:
        """Number of batches per epoch"""
        return math.ceil(len(self.dataset) / self.batch_size)
