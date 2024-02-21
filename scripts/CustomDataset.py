import pandas as pd

import SimpleITK as sitk

import torch

from pathlib import Path

from torch.utils.data import Dataset, Subset

import torchvision.transforms as transforms
from configparser import ConfigParser
from matplotlib import pyplot as plt


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

        return image, label

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

    def get_sequence_length_freq(self):

        n_frames = []
        parser = ConfigParser()

        for idx in range(len(self)):

            patient_file, _ = self._get_patient_data(idx)
            patient_dir = patient_file[:11]
            view = patient_file[12:15]
            info_file = Path.cwd().joinpath(
                "data", "database_nifti", patient_dir, f"Info_{view}.cfg"
            )
            with open(info_file) as stream:
                parser.read_string("[info]\n" + stream.read())
                n_frames.append(parser.getint("info", "NbFrame"))

        plt.hist(n_frames, bins=max(n_frames))
        plt.xlabel("Number of Frames")
        plt.ylabel("Frequency")
        plt.savefig("number_of_images.png", bbox_inches="tight")
