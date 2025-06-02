import os
import torch
import torch.nn as nn
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from torch.optim import AdamW, SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings
from data_class import CheXpertDataset


class CheXpertSwinV2Model(nn.Module):
    def __init__(self, img_size, pretrained_weights, gpu, lr, optim):
        super().__init__()

        # Handle image size
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        # Initialize base model
        if pretrained_weights == "ImageNet_1k":
            weights = Swin_V2_B_Weights.IMAGENET1K_V1
            self.base_model = swin_v2_b(weights=weights)
            self.transform = weights.transforms()  # Use pretrained transforms
            print("Loaded ImageNet-1k weights.")
        elif pretrained_weights == "Random":
            self.base_model = swin_v2_b(weights=None)
            self.transform = self.get_default_transform(img_size)
            print("Initialized with random weights.")
        else:  # Custom weights
            self.base_model = swin_v2_b(weights=None)
            self.transform = self.get_default_transform(img_size)
            try:
                state_dict = torch.load(pretrained_weights)
                self.base_model.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights from {pretrained_weights}")
            except Exception as e:
                raise RuntimeError(f"Error loading weights: {e}")

        # Replace head
        in_features = self.base_model.head.in_features
        self.base_model.head = nn.Sequential(
            nn.Linear(in_features, int(in_features / 3)),
            nn.Dropout(0.1),
            nn.Linear(int(in_features / 3), int(in_features / 3)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_features / 3), 14),
            nn.Sigmoid() # Sigmoid for multi-label classification
        )

        # Device setup
        self.device = torch.device(f"cuda:{gpu}" if gpu is not None and torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Using device: {self.device}")

        # Optimizer (after full model setup)
        if optim == "adamw":
            self.optimizer = AdamW(self.parameters(), lr=lr)
        elif optim == "adam":
            self.optimizer = Adam(self.parameters(), lr=lr)
        elif optim == "sgd":
            self.optimizer = SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError("Optimizer must be 'adamw', 'adam', or 'sgd'")

    def get_default_transform(self, img_size):
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.base_model(x)