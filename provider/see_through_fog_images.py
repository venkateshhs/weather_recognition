import pathlib
import typing as T
import numpy as np
import PIL
import torchvision

from util.utils import get_base_path, get_timestamp


class Frame:
    def __init__(self, frame_type, timestamp, image):
        self.frame_type = frame_type
        self.timestamp = timestamp
        self.image = image


class Image:
    def __init__(self, data):
        self.data = data


class SeeingThroughFogImageDataset:

    def __init__(self, split: str, base_path: T.Optional[str] = "", normalize: bool = False,
                 transforms=None):
        """
        :param split: Which split to load (train, val, test)
        :param normalize: Whether to normalize/standardize images before returning them in __getitem__()
        """
        self.transforms = transforms
        self.normalize = normalize
        self.file_names = []
        self.timestamps = []
        directory_paths, file_paths = get_base_path(split,
                                                    base_path=base_path, assume_existing_files_only=True)
        self.img_filepaths = sorted(file_paths)
        self.file_names = [path.name for path in file_paths]

        for file_name in self.file_names:
            self.timestamps.append(get_timestamp(file_name))

    def get_data_timestamps(self) -> T.Sequence[float]:
        return self.timestamps

    def __getitem__(self, index: int) -> Frame:
        img = PIL.Image.open(self.img_filepaths[index])
        img_tensor = torchvision.transforms.ToTensor()(img)
        if self.transforms:
            img_tensor = self.transforms(img_tensor)
        if self.normalize:
            normalize = torchvision.transforms.Normalize(mean=[0.364, 0.349, 0.331], std=[0.183, 0.183, 0.184])
            img_tensor = normalize(img_tensor)
        frame = Frame("CAMERA", self.timestamps[index], Image(img_tensor))
        return frame


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = SeeingThroughFogImageDataset('train')
    frame = dataset[0]
    img_tensor = frame.image.data
    # Convert the tensor to a NumPy array and transpose the dimensions for plotting
    # The tensor is in CxHxW format, but matplotlib expects HxWxC format
    img_array = img_tensor.numpy().transpose(1, 2, 0)
    # Plot the image
    plt.imshow(img_array)
    plt.show()
