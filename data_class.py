from torch.utils.data import Dataset
import numpy as np
import PIL
import PIL.Image as Image
import csv
import torch


class CheXpertDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = []  # List to hold image file paths
        self.labels = []       # List to hold labels

        self.chexpert_mappings = {
            0: "No Finding",
            1: "Enlarged Cardiomediastinum",
            2: "Cardiomegaly",
            3: "Lung Opacity",
            4: "Lung Lesion",
        }

        with open(dataset_path, "r") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                self.image_paths.append(row[0])
                row_label = []
                # print(row)
                for i in range(5, len(row)):
                    if row[i] == "1.0":
                        row_label.append(1)
                    elif row[i] == "-1.0":
                        row_label.append(-1)
                    else:
                        row_label.append(0)
                self.labels.append(torch.tensor(row_label))


    @staticmethod
    def _augment_image(image):
        # Normalize the image
        image =  np.array(image, dtype=np.float32) / 255.0
        return image

    def decode_label(self, label):
        # Convert label indices to human-readable format
        return [self.chexpert_mappings[i] for i in range(len(label)) if label[i] == 1]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = PIL.Image.open(image_path).convert('RGB').resize((224, 224))  # Resize to 224x224

        image = torch.tensor(self._augment_image(image).transpose(2, 0, 1).astype('float32'), dtype=torch.float32)

        return image, label

