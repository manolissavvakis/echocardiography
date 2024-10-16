import pandas as pd
from pathlib import Path
import numpy as np
import glob
import shutil


def split_data(file_list):
    
    directories = [Path.cwd() / "train_data", Path.cwd() / "test_data", Path.cwd() / "val_data"]
    video_dir = Path.cwd() / "Videos"
    df = None

    for dir in directories:
        if not dir.is_dir():        
            dir.mkdir()
            print(str(dir) + ' has been created.')
            df = pd.read_csv(file_list, delimiter=",") if not df else None
            split = dir.stem[:-5]
            
            split_df = df.loc[df["Split"] == split.upper()]
            split_df.to_csv(f"{split}_files.csv", index=False)
                    
            for video in split_df["FileName"].values:
                video_path = f"{video_dir/video}.avi"
                if glob.glob(video_path):
                    shutil.copy(video_path, dir)
            print("Copying files in directory " + str(dir) + " is completed.")


def csv_with_labels():

    file_list = Path.cwd().joinpath("FileListWithLabels.csv")
    if not file_list.is_file():
        df = pd.read_csv("FileList.csv", delimiter=",")

        # If EF <= 45., patient has pathological risk.
        df["Label"] = np.where(df["EF"] <= 45.0, 1, 0)

        # Save the new dataframe.
        df.to_csv("FileListWithLabels.csv", index=False)

    return file_list
