import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='albumentations')

class AlbumentationsCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Convert PIL Image to numpy array for albumentations
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
            
        return image, label

class CIFAR10DataModule:
    def __init__(self, config):
        self.config = config
        # Calculate mean pixel value for CoarseDropout
        self.mean_pixels = [int(m * 255) for m in [0.4914, 0.4822, 0.4465]]
        
        self.train_transform = A.Compose([
            A.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),

            A.HorizontalFlip(p=0.5),

            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=9,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=1,
                max_height=8,
                max_width=8,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=self.mean_pixels,  # Using mean pixel values for filling
                mask_fill_value=None,
                p=0.5
            ),

            A.RandomBrightnessContrast(p=0.2),

            ToTensorV2()
        ])
        
        self.test_transform = A.Compose([
            A.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
            ToTensorV2()
        ])

    def setup(self):
        self.train_dataset = AlbumentationsCIFAR10(
            root=self.config.DATA_DIR,
            train=True,
            transform=self.train_transform
        )
        
        self.test_dataset = AlbumentationsCIFAR10(
            root=self.config.DATA_DIR,
            train=False,
            transform=self.test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )