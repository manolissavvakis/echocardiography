import pandas as pd
import SimpleITK as sitk
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from configparser import ConfigParser
from matplotlib import pyplot as plt
from collections import Counter
import random
import numpy as np


class CustomSequenceDataset(Dataset):
    """
    Custom Sequence Dataset. Load CAMUS dataset information properly.

    :param path: Path to each dataset fold.
    :param transform: Whether to apply transforms to a sequence before it is returned.
    :param common_sequence: The number of the most common sequences to be retained for training.
        Default value is 1 (keep only the most common).
    """

    def __init__(self, path: str, transform: bool, common_sequences=1):

        self.path = path

        def set_seq(patient: str):
            return f"{patient[:15]}_half_sequence{patient[18:]}"

        self.training_data = pd.read_csv(
            Path.joinpath(self.path, "train.csv"),
            usecols=["patient_id", "pathological_risk_label"],
        )
        self.training_data["patient_id"] = self.training_data["patient_id"].apply(
            set_seq
        )

        self.validation_data = pd.read_csv(
            Path.joinpath(self.path, "val.csv"),
            usecols=["patient_id", "pathological_risk_label"],
        )
        self.validation_data["patient_id"] = self.validation_data["patient_id"].apply(
            set_seq
        )

        self.most_frequent_lengths = self._get_most_frequent_length(common_sequences)
        self.transform = transform

    def __len__(self):

        return len(self.training_data) + len(self.validation_data)

    def _get_data_source(self, idx):
        """
        Get the correct data source.
        """

        if idx < len(self.training_data):

            return self.training_data

        else:

            return self.validation_data

    def _get_correct_idx(self, idx):
        """
        Get the correct idx for an item.
        """

        return idx if idx < len(self.training_data) else idx - len(self.training_data)

    def __getitem__(self, idx):

        patient_file, label = self._get_patient_data(idx)

        patient_dir = patient_file[:11]

        image = sitk.ReadImage(
            Path.cwd().joinpath("data", "database_nifti", patient_dir, patient_file)
        )

        image = sitk.GetArrayFromImage(image)
        image = image[None]

        if len(self.most_frequent_lengths) > 1:
            sequence_length = image.shape[-3]
            diff = sequence_length - self.most_frequent_lengths[0]
            if diff > 0:
                imgs_to_delete = random.sample(range(1, sequence_length), diff)
                image = np.delete(image, imgs_to_delete, axis=1)
            else:
                imgs_to_fill = np.zeros(
                    [1, abs(diff), image.shape[-2], image.shape[-1]], dtype=np.float32
                )
                image = np.concatenate(
                    (image[:, :-1], imgs_to_fill, image[None, :, -1]), axis=1
                )

        if self.transform:

            image, label = self.transformations(image, label)

        return image, label

    def _get_patient_data(self, idx):
        """
        :return: the name of the sequence's file and its label.
        """

        data_source = self._get_data_source(idx)

        idx = self._get_correct_idx(idx)

        patient_file, label = data_source.iloc[idx]

        return patient_file, label

    def transformations(self, image, label):
        """
        Apply transforms to a sequence. Transforms include:
            - Resize each image to (256, 256)
            - Normalise each image's pixels to [0., 1.]
            - return sequence and label as tensors
        """

        resize_tensor = transforms.Resize((256, 256), antialias=True)

        def normalise(sample):

            dims = sample.size()
            sample = sample.view(dims[0], -1)
            sample -= sample.min(1, keepdim=True)[0]
            sample /= sample.max(1, keepdim=True)[0]
            sample = sample.view(dims)

            return sample

        image = torch.Tensor(image)
        image = normalise(resize_tensor(image))

        label = torch.tensor(label)

        return image, label

    def get_sequence_length(self, idx):
        """
        :return: the length of the sequence in index 'idx'.
        """

        patient_file, _ = self._get_patient_data(idx)
        patient_dir = patient_file[:11]
        view = patient_file[12:15]
        info_file = Path.cwd().joinpath(
            "data", "database_nifti", patient_dir, f"Info_{view}.cfg"
        )

        parser = ConfigParser()
        with open(info_file) as stream:
            # Read the info file concerning the sequence.
            parser.read_string("[info]\n" + stream.read())
            return parser.getint("info", "NbFrame")

    def _get_most_frequent_length(self, n_most_common=1):
        """
        Get the 'n_most_common' most common sequence lengths found in the dataset.
        """

        n_frames = []

        for idx in range(len(self)):
            n_frames.append(self.get_sequence_length(idx))

        most_common_list = Counter(n_frames).most_common(n_most_common)
        most_frequent_length_list = []
        for idx in range(n_most_common):
            most_frequent_length_list.append(most_common_list[idx][0])

        # Plot a histogram with the lengths found.
        plt.hist(n_frames, bins=most_frequent_length_list[0])
        plt.xlabel("Number of Frames")
        plt.ylabel("Frequency")
        plt.savefig("number_of_images.png", bbox_inches="tight")

        return most_frequent_length_list

    def create_most_frequent_length_subset(self):
        """
        :return: a CustomSequenceDataset including only the sequences whose length is in the most common.
        """
        training_indices = [
            idx
            for idx in range(len(self.training_data))
            if self.get_sequence_length(idx) in self.most_frequent_lengths
        ]
        validation_indices = [
            self._get_correct_idx(idx)
            for idx in range(len(self.training_data), len(self))
            if self.get_sequence_length(idx) in self.most_frequent_lengths
        ]

        self.training_data = self.training_data.iloc[training_indices, :].reset_index(
            drop=True
        )
        self.validation_data = self.validation_data.iloc[
            validation_indices, :
        ].reset_index(drop=True)
        return self


class CustomValDataset(Dataset):
    """
    Custom Validation Dataset. Keeps only unique patients.

    :param dataset: CustomSequenceDataset used in training phase.
    """

    def __init__(self, dataset: CustomSequenceDataset):

        # To keep unique patients, filter through first 11 characters of 'patient_id'.
        self.validation_data = dataset.validation_data
        self.validation_data["view"] = self.validation_data["patient_id"].str[11:]
        self.validation_data["patient_id"] = self.validation_data["patient_id"].str[:11]
        self.validation_data = self.validation_data.drop_duplicates(subset="patient_id")
        self.validation_data["patient_id"] = (
            self.validation_data.loc[:, "patient_id"]
            + self.validation_data.loc[:, "view"]
        )
        self.validation_data = self.validation_data.drop("view", axis=1).reset_index(
            drop=True
        )

        self.most_frequent_lengths = dataset.most_frequent_lengths
        self.transform = dataset.transform

    def __len__(self):

        return len(self.validation_data)

    def __getitem__(self, idx):
        """
        Get a batch which which includes the 2 sequences of the patient (2CH and 4CH) and
            his/her corresponding label. If one of the sequences has not the most common length, delete random
            images of the sequence (except the first and last frame) or fill it with black images, until it reaches
            that length. Transforms applied to the batch are the same as those of the training dataset.

        :return: a tensor of shape (1, 2, D, H, W) and its label.
        """

        patient_file, label = self.validation_data.iloc[idx]

        patient_dir = patient_file[:11]

        images_list = []

        for view in ["2CH", "4CH"]:
            image = sitk.ReadImage(
                [
                    Path.cwd().joinpath(
                        "data",
                        "database_nifti",
                        patient_dir,
                        f"{patient_dir}_{view}_half_sequence.nii.gz",
                    )
                ]
            )
            image = sitk.GetArrayFromImage(image)
            # image's shape is: (N, D, H, W)

            sequence_length = image.shape[-3]

            if sequence_length != self.most_frequent_lengths[0]:

                diff = sequence_length - self.most_frequent_lengths[0]
                if diff > 0:
                    imgs_to_delete = random.sample(range(1, sequence_length), diff)
                    image = np.delete(image, imgs_to_delete, axis=1)
                else:
                    imgs_to_fill = np.zeros(
                        [1, abs(diff), image.shape[-2], image.shape[-1]],
                        dtype=np.float32,
                    )
                    image = np.concatenate(
                        (image[:, :-1], imgs_to_fill, image[None, :, -1]), axis=1
                    )

            if self.transform:
                image, label = self.transformations(image, label)

            images_list.append(image)

        images_to_eval = torch.cat(images_list)
        images_to_eval = images_to_eval[:, None]

        return images_to_eval, label
