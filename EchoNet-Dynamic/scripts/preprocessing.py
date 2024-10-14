import pandas as pd
from pathlib import Path
import numpy as np
import glob
import shutil


def split_data(file_list):
    
    directories = [Path.cwd() / "train_data", Path.cwd() / "test_data", Path.cwd() / "validation_data"]

    if not (train_dir.is_dir() or val_dir.is_dir() or test_dir.is_dir()):
        df = pd.read_csv(file_list, delimiter=",")

        train_df = df.loc[df["Split"] == "TRAIN"]
        train_df.to_csv("train_files.csv", index=False)

        val_df = df.loc[df["Split"] == "VAL"]
        val_df.to_csv("val_files.csv", index=False)

        test_df = df.loc[df["Split"] == "TEST"]
        test_df.to_csv("test_files.csv", index=False)

        for dir in directories:
            dir.mkdir(exist_ok=True)

        video_dir = Path.cwd() / "Videos"
        
        for video in train_df["FileName"].values:
            video_path = f"{video_dir/video}.avi"
            if glob.glob(video_path):
                shutil.copy(video_path, directories[0])
                
        print("Copying files in directory " + str(directories[0]) + " is completed.")
        
        for video in test_df["FileName"].values:
            video_path = f"{video_dir/video}.avi"
            if glob.glob(video_path):
                shutil.copy(video_path, directories[1])
        print("Copying files in directory " + str(directories[1]) + " is completed.")
        
        for video in val_df["FileName"].values:
            video_path = f"{video_dir/video}.avi"
            if glob.glob(video_path):
                shutil.copy(video_path, directories[2])
        print("Copying files in directory " + str(directories[2]) + " is completed.")


def csv_with_labels():

    file_list = Path.cwd().joinpath("FileListWithLabels.csv")
    if not file_list.is_file():
        df = pd.read_csv("FileList.csv", delimiter=",")

        # If EF <= 45., patient has pathological risk.
        df["Label"] = np.where(df["EF"] <= 45.0, 1, 0)

        # Save the new dataframe.
        df.to_csv("FileListWithLabels.csv", index=False)

    return file_list
