import pathlib
import typing as T

from datetime import datetime, timezone, timedelta
import yaml
import os


def get_timestamp(date_str: str):
    # date_str = 2018-02-03_20-48-35_00400.png
    date_str_parsed, fractional_seconds_str = date_str.rsplit('_', 1)
    fractional_seconds = int(fractional_seconds_str.split('.')[0]) / 10
    dt = datetime.strptime(date_str_parsed, "%Y-%m-%d_%H-%M-%S")
    dt_with_fraction = dt + timedelta(seconds=fractional_seconds)
    dt_utc = dt_with_fraction.replace(tzinfo=timezone.utc)
    return dt_utc.timestamp()


def get_base_path(split: T.Union[str, T.Iterable[str]], base_path: T.Optional[str] = "",
                  assume_existing_files_only=False):
    dataset_files_config = r'C:\Users\venkatesh\Desktop\Personal Projects\weather_recognition\resources\dense_images_split.yaml'
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
                raise FileNotFoundError(f"Could not find file in , {split}: {df_path}")
            if df_path.is_dir():
                directory_paths.append(df_path)
            elif df_path.is_file():
                file_paths.append(df_path)
            else:
                raise RuntimeError(f"Unknown type of file: {df_path}")
    return directory_paths, file_paths
