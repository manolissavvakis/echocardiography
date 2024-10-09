import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
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

    def __init__(self, data: str, videos_dir: str, transform=True):

        self.file_list = pd.read_csv(data)

        self.videos_dir = videos_dir
        self.most_frequent_length = self._get_most_frequent_length()
        self.transform = transform

    def __len__(self):

        return len(self.file_list)

    def __getitem__(self, idx):

        patient_file, label, fps, num_frames = self.file_list.loc[
            idx, ["FileName", "Label", "FPS", "NumberOfFrames"]
        ]
        video_path = f"{self.videos_dir}/{patient_file}.avi"

        sequence = self.extract_frames(video_path, fps, num_frames)

        # sequence shape = (C, D, H, W)
        sequence_length = sequence.shape[-3]
        diff = sequence_length - self.most_frequent_length
        if diff > 0:
            imgs_to_delete = random.sample(range(1, sequence_length), diff)
            sequence = np.delete(sequence, imgs_to_delete, axis=1)
        else:
            imgs_to_fill = np.ones(
                [abs(diff)] + list(sequence.shape[-2:]), dtype=np.float32
            )
            sequence = np.concatenate(
                (sequence[:, :-1], imgs_to_fill[None], sequence[:, -1:]), axis=1
            )

        if self.transform:
            sequence, label = self.transformations(sequence, label)

        return sequence, label

    def transformations(self, sequence, label):
        """
        Apply transforms to a sequence. Transforms include:
            - Resize each image to (64, 64)
            - Normalise each image's pixels to [0., 1.]
            - return sequence and label as tensors
        """

        resize_tensor = transforms.Resize((64, 64), antialias=True)

        def normalise(sample):

            dims = sample.size()
            sample = sample.view(dims[0], -1)
            sample -= sample.min(1, keepdim=True)[0]
            sample /= sample.max(1, keepdim=True)[0]
            sample = sample.view(dims)

            return sample

        sequence = torch.Tensor(sequence)
        sequence = normalise(resize_tensor(sequence))
        # sequence = transforms.Normalize(0.5, 0.5)

        label = torch.tensor(label)

        return sequence, label

    def _get_most_frequent_length(self):
        """
        Get the 'n_most_common' most common sequence lengths found in the dataset.
        """

        n_frames = []

        for idx in range(len(self)):
            n_frames.append(self.file_list.iloc[idx]["NumberOfFrames"])

        most_common_length = Counter(n_frames).most_common(1)[0][0]

        return most_common_length

    def extract_frames(self, video_file, fps, num_frames):
        """
        Extract a specified number of frames from the video at the given frame rate.

        :param video_file: Path to the video file.
        :param num_frames: Number of frames to extract.
        :param fps: Desired frames per second.
        :return: A numpy array of extracted frames.
        """

        cap = cv2.VideoCapture(video_file)
        frames = []
        frame_count = 0
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)

        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            frame_count += 1

        cap.release()

        # Shape is (D, H, W)
        frames = np.array(frames)

        # Return shape is (C, D, H, W)
        return np.expand_dims(frames, axis=0)
