from preprocessing import *
import pandas as pd
import cv2
from torchvision.transforms import v2
from pathlib import Path
from PIL import Image
import csv
from torchvision import tv_tensors
import torchio as tio
import torch


import matplotlib.pyplot as plt


class VideoAugmentor:
    def __init__(
        self,
        input_csv: str,
        videos_dir: str,
        split: str,
        target_label: int,
        target_size: int,
    ):
        self.input_csv = input_csv
        self.videos_dir = videos_dir
        self.target_size = target_size

        if split is not ("train" or "validation" or "test"):
            raise Exception("Split must be train, validation or test")

        # Filter videos with the target label (e.g., label 1) and split selected.
        self.target_class_videos = self.input_csv[
            (self.input_csv["Label"] == target_label)
            & (self.input_csv["Split"] == split.upper())
        ]

        # Define the torchio transforms for grayscale images
        self.transform = tio.transforms.Compose(
            [
                tio.transforms.RandomFlip(axes=(1, 2), flip_probability=0.5, p=0.25),
                tio.transforms.RandomAffine(
                    scales=(0.9, 1.1), translation=(-20, 20), p=0.25
                ),
                tio.transforms.RandomSpike(num_spikes=2, p=0.25),
                tio.transforms.RandomGhosting(num_ghosts=(1, 3), axes=(1, 2), p=0.25),
                tio.transforms.RandomBlur(std=(0, 2), p=0.25),
                tio.transforms.RandomBiasField(p=0.25),
                tio.transforms.RandomNoise(p=0.25),
            ]
        )

    def augment_and_save(self):
        count_augmented = 0
        while len(self.target_class_videos) + count_augmented < self.target_size:

            # Select a random video
            video_row = self.target_class_videos.sample().iloc[0]
            id = video_row.name
            fps, num_frames = video_row[["FPS", "NumberOfFrames"]]
            video_file = f"{self.videos_dir}/{video_row['FileName']}.avi"
            augmented_video = self.apply_augmentations(video_file, fps, num_frames)

            # Save the augmented video in the general videos folder and csv files.
            output_file_name = f"{video_row['FileName']}_augmented_{count_augmented}"
            self.save_video(
                augmented_video.squeeze(axis=0),
                str(self.videos_dir / output_file_name) + ".avi",
                fps,
            )
            self.save_csv(
                output_file_name, id, self.videos_dir.parent / "FileListWithLabels.csv"
            )
            self.save_csv(output_file_name, id, self.videos_dir.parent / "FileList.csv")

            count_augmented += 1

            print(f"Saved augmented video {output_file_name}")

    def apply_augmentations(self, video_file, fps, num_frames):

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

                # Convert to a PIL Image for torchvision compatibility (mode 'L' for grayscale)
                # frame = Image.fromarray(frame, mode="L")

                # Apply transformations using torchvision
                # frame = self.transform(frame)

                # Convert back to NumPy array
                # frame = np.array(frame)
                frames.append(frame)

            frame_count += 1

        cap.release()

        # Convert to tv_tensor for torchvision.transforms.v2 compatibility
        frames = np.asarray(frames, dtype=np.float32)
        frames = tv_tensors.Video(np.expand_dims(frames, axis=0))

        # Apply transformations using torchvision
        # torch.permute(frames, (0, 2, 3, 1))
        # frames = self.transform(torch.permute(frames, (0, 2, 3, 1)))
        frames = self.transform(torch.permute(frames, (0, 2, 3, 1)))

        # Convert back to numpy array
        frames = torch.permute(frames, (0, 3, 1, 2)).numpy()
        return frames

    def save_video(self, frames, output_path, fps):
        if len(frames) == 0:
            return

        height, width = frames.shape[-2:]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height), False)

        for frame in frames:
            out.write(frame.astype(np.uint8))

        out.release()

    def save_csv(self, video_file_to_save, index, output_csv_file):

        with open(output_csv_file, "a", newline="") as file_list:

            writer_object = csv.writer(file_list, delimiter=",")

            row = self.input_csv.iloc[index]
            writer_object.writerow(row.replace(row["FileName"], video_file_to_save))

            # Close the file object
            file_list.close()


def main():

    input_csv = csv_with_labels()

    videos_dir = Path.cwd() / "Videos"  # Directory containing .avi files

    # Load the CSV file to find the count of each class
    file_list = pd.read_csv(input_csv)
    class_0_count = len(file_list[file_list["Label"] == 0])
    class_1_count = len(file_list[file_list["Label"] == 1])

    # Target number of videos for class 1 to balance with class 0
    target_size = class_0_count
    split = "train"

    augmentor = VideoAugmentor(
        file_list, videos_dir, split, target_label=1, target_size=target_size
    )
    augmentor.augment_and_save()


if __name__ == "__main__":
    main()
