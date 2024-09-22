import pandas as pd
from pathlib import Path
import numpy as np
import glob
import shutil


def split_data(file_list):
    train_dir = Path.cwd() / "train_data"
    val_dir = Path.cwd() / "validation_data"
    test_dir = Path.cwd() / "test_data"

    if not (train_dir.is_dir() or val_dir.is_dir() or test_dir.is_dir()):
        df = pd.read_csv(file_list, delimiter=",")

        train_df = df.loc[df["Split"] == "TRAIN"]
        train_df.to_csv("train_files.csv", index=False)

        val_df = df.loc[df["Split"] == "VAL"]
        val_df.to_csv("val_files.csv", index=False)

        test_df = df.loc[df["Split"] == "TEST"]
        test_df.to_csv("test_files.csv", index=False)

        for dir in [train_dir, val_dir, test_dir]:
            Path(dir).mkdir(exist_ok=True)

            video_dir = Path.cwd() / "Videos"
            for video in train_df["FileName"].values:
                video_path = f"{video_dir/video}.avi"
                if glob.glob(video_path):
                    shutil.copy(video_path, train_dir)

            for video in test_df["FileName"].values:
                video_path = f"{video_dir/video}.avi"
                if glob.glob(video_path):
                    shutil.copy(video_path, test_dir)

            for video in val_df["FileName"].values:
                video_path = f"{video_dir/video}.avi"
                if glob.glob(video_path):
                    shutil.copy(video_path, val_dir)

        print("Copying files is completed.")


def csv_with_labels():

    file_list = Path.cwd().joinpath("FileListWithLabels.csv")
    if not file_list.is_file():
        df = pd.read_csv("FileList.csv", delimiter=",")

        # If EF <= 45., patient has pathological risk.
        df["Label"] = np.where(df["EF"] <= 45.0, 1, 0)

        # Save the new dataframe.
        df.to_csv("FileListWithLabels.csv", index=False)

    return file_list
