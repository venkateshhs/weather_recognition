import pathlib
import typing as T
import numpy as np
import PIL
import torch
import torchvision
import yaml

import os
from datetime import datetime, timezone, timedelta


def get_base_path(directory: str, split: T.Union[str, T.Iterable[str]], base_path: T.Optional[str] = "",
                  assume_existing_files_only=False):
    dataset_files_config = r'C:\Users\venkatesh\Desktop\Projects\weather-recognition\resources\dense_images_split.yaml'
    with open(dataset_files_config) as fo:
        dfs = yaml.safe_load(fo)
    directory_paths, file_paths = [], []
    if isinstance(split, str):
        split = [split]
    for s in split:
        split_dfs = dfs[s]
        for df_path in split_dfs:
            df_path = df_path.replace("\\", "/")
            df_path = pathlib.Path(df_path)
            if not df_path.is_absolute():
                df_path = pathlib.Path(os.path.join(base_path, df_path))
            if assume_existing_files_only:
                file_paths.append(df_path)
                continue
            if not df_path.exists():
                raise FileNotFoundError(f"Could not find file in {directory}, {split}: {df_path}")
            if df_path.is_dir():
                directory_paths.append(df_path)
            elif df_path.is_file():
                file_paths.append(df_path)
            else:
                raise RuntimeError(f"Unknown type of file: {df_path}")
    return directory_paths, file_paths


def get_timestamp(date_str: str):
    # date_str = 2018-02-03_20-48-35_00400.png
    date_str_parsed, fractional_seconds_str = date_str.rsplit('_', 1)
    fractional_seconds = int(fractional_seconds_str.split('.')[0]) / 10
    dt = datetime.strptime(date_str_parsed, "%Y-%m-%d_%H-%M-%S")
    dt_with_fraction = dt + timedelta(seconds=fractional_seconds)
    dt_utc = dt_with_fraction.replace(tzinfo=timezone.utc)
    return dt_utc.timestamp()


class Frame:
    def __init__(self, frame_type, timestamp, image):
        self.frame_type = frame_type
        self.timestamp = timestamp
        self.image = image


class Image:
    def __init__(self, data):
        self.data = data


class SeeingThroughFogImageDataset:

    def __init__(self, directory_path: str, split: str, base_path: T.Optional[str] = "", normalize: bool = False,
                 transforms=None):
        """
        :param directory_path: Path to the directory of the SeeingThroughFog dataset where the image files are contained
        :param split: Which split to load (train, val, test)
        :param normalize: Whether to normalize/standardize images before returning them in __getitem__()
        """
        self.transforms = transforms
        self.directory_path = pathlib.Path(directory_path)
        self.file_names = []
        self.timestamps = []
        directory_paths, file_paths = get_base_path(directory_path, split,
                                                    base_path=base_path)
        self.img_filepaths = sorted(file_paths)
        self.file_names = [path.name for path in file_paths]

        for file_name in self.file_names:
            self.timestamps.append(get_timestamp(file_name))

    def get_data_timestamps(self) -> T.Sequence[float]:
        return self.timestamps

    def __getitem__(self, index: int) -> Frame:
        img = PIL.Image.open(self.img_filepaths[index])
        img_array = (np.asarray(img, dtype="float32") / 255)
        img_tensor = torchvision.transforms.ToTensor()(img_array)
        if self.transforms:
            img_tensor = self.transforms(img_tensor)
        frame = Frame("CAMERA", self.timestamps[index], Image(img_tensor))
        return frame


if __name__ == '__main__':
    dataset_path = r''
    SeeingThroughFogImageDataset(dataset_path, 'test')
