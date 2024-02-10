import pandas as pd
import SimpleITK as sitk
import torch
from pathlib import Path
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms


class CustomImageDataset(Dataset):
    def __init__(self, path: str, transform: bool):
        self.path = path
        self.training_data = pd.read_csv(
            Path.joinpath(self.path, "train.csv"),
            usecols=["patient_id", "pathological_risk_label"],
        )
        self.validation_data = pd.read_csv(
            Path.joinpath(self.path, "val.csv"),
            usecols=["patient_id", "pathological_risk_label"],
        )
        self.transform = transform

    def __len__(self):
        return len(self.training_data) + len(self.validation_data)

    def __getitem__(self, idx):
        if idx < len(self.training_data):
            data_source = self.training_data
        else:
            data_source = self.validation_data
            idx -= len(self.training_data)

        patient_file, label = data_source.iloc[idx]
        patient_dir = patient_file[:11]
        image = sitk.ReadImage(
            Path.cwd().joinpath("data", "database_nifti", patient_dir, patient_file)
        )
        image = sitk.GetArrayFromImage(image)
        if self.transform:
            image, label = self.transformations(image, label)
        return image, label

    def split_data(self):
        training_subset = Subset(self, list(range(len(self.training_data))))
        validation_subset = Subset(self, list(range(len(self.validation_data))))
        return training_subset, validation_subset

    def transformations(self, image, label):

        tensor_resize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True),
            ]
        )

        def normalise(sample):
            """
            Normalise images to [0., 1.].
            """
            dims = sample.size()
            sample = sample.view(sample.size(0), -1)
            sample -= sample.min(1, keepdim=True)[0]
            sample /= sample.max(1, keepdim=True)[0]
            sample = sample.view(dims)
            return sample

        image = normalise(tensor_resize(image))
        label = torch.tensor(label)

        return image, label
