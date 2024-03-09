import pandas as pd

import SimpleITK as sitk

import torch

from pathlib import Path

from torch.utils.data import Dataset, Subset

import torchvision.transforms as transforms
from configparser import ConfigParser
from matplotlib import pyplot as plt
from collections import Counter


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

    def _get_data_source(self, idx):

        if idx < len(self.training_data):

            return self.training_data

        else:

            return self.validation_data

    def _get_correct_idx(self, idx):

        return idx if idx < len(self.training_data) else idx - len(self.training_data)

    def __getitem__(self, idx):

        patient_file, label = self._get_patient_data(idx)

        patient_dir = patient_file[:11]

        image = sitk.ReadImage(
            Path.cwd().joinpath("data", "database_nifti", patient_dir, patient_file)
        )

        image = sitk.GetArrayFromImage(image)

        if self.transform:

            image, label = self.transformations(image, label)

        return image, label, patient_dir

    def _get_patient_data(self, idx):

        data_source = self._get_data_source(idx)

        idx = self._get_correct_idx(idx)

        patient_file, label = data_source.iloc[idx]

        return patient_file, label

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

    def get_sequence_length(self, idx):
        patient_file, _ = self._get_patient_data(idx)
        patient_dir = patient_file[:11]
        view = patient_file[12:15]
        info_file = Path.cwd().joinpath("data", "database_nifti", patient_dir, f"Info_{view}.cfg")
        
        parser = ConfigParser()
        with open(info_file) as stream:
            parser.read_string("[info]\n" + stream.read())
            return parser.getint("info", "NbFrame")

    def get_most_frequent_length(self):

        n_frames = []

        for idx in range(len(self)):
            n_frames.append(self.get_sequence_length(idx))

        most_frequent_length = Counter(n_frames).most_common(1)[0][0]

        plt.hist(n_frames, bins=most_frequent_length)
        plt.xlabel("Number of Frames")
        plt.ylabel("Frequency")
        plt.savefig("number_of_images.png", bbox_inches="tight")

        return most_frequent_length

    def create_most_frequent_length_subset(self):
        most_frequent_length = self.get_most_frequent_length()
        subset_indices = [idx for idx in range(len(self)) if self.get_sequence_length(idx) == most_frequent_length]
        return Subset(self, subset_indices)