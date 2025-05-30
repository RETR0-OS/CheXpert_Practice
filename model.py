import os
import torch
from torchvision.models.swin_transformer import swin_v2_b
from torchvision.models import Swin_V2_B_Weights
from torch.optim import AdamW, SGD, Adam
from torch.utils.data import DataLoader
from data_class import CheXpertDataset
import warnings
import torch.nn as nn

class CheXpertSwinV2Model(swin_v2_b):
    def __init__(self, img_size, batch_size, epochs, pretrained_weights, save_model_dir, logs_dir, gpu, lr, optim, dataset_path, workers=2, *args, **kwargs):
        # Load model with pretrained weights if specified
        if pretrained_weights == "ImageNet_1k":
            self.model = super().__init__(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
            print("Initialized model with ImageNet-1k pretrained weights.")
        elif pretrained_weights == "Random":
            self.model = super().__init__(weights=None)
            print("Initialized model with random weights.")
        else:
            try:
                weights_path = os.path.abspath(pretrained_weights)
                if os.path.exists(weights_path):
                    self.model = super().__init__(weights=None)
                    self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                    print(f"Initialized model with pretrained weights from {weights_path}.")
                else:
                    raise FileNotFoundError(f"Pretrained weights file not found: {weights_path}")
            except Exception as e:
                raise ValueError(f"Please check your pretrained weights path. Error loading pretrained weights: {e}")

        self.val_dataset = None
        self.train_dataset = None
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_model_dir = save_model_dir
        self.logs_dir = logs_dir
        self.gpu = gpu
        self.lr = lr
        self.optim = optim
        self.dataset_path = dataset_path
        self.workers = workers

        # Set device
        if gpu is not None:
            if not torch.cuda.is_available():
                warnings.warn("CUDA is not available. Please check your GPU setup. Defaulting to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{gpu}")
                print("Using GPU:", self.device)
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.classification_head = nn.Sequential(nn.Linear(self.model.head.in_features, int(self.model.head.in_features / 3)), nn.ReLU(inplace=True), nn.Linear(int(self.model.head.in_features / 3), 14))


        # Move model to device
        self.model.to(self.device)

        # Set optimizer
        if optim == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=lr)
        elif optim == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=lr)
        elif optim == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer option.")

    def load_dataset(self):
        self.train_dataset = DataLoader(CheXpertDataset(self.dataset_path+"/train.csv", transform=None), num_workers=self.workers, batch_size=self.batch_size, shuffle=True)
        self.val_dataset = DataLoader(CheXpertDataset(self.dataset_path+"/valid.csv", transform=None), num_workers=self.workers, batch_size=self.batch_size, shuffle=False)

