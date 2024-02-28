import json
import pathlib
import typing as T
import numpy as np
import PIL
import torch
import torchvision
import logging
import torchvision.transforms as transforms
from util.utils import get_base_path, get_timestamp


def label_to_int(label):
    label_mapping = {
        'dense_fog': 0,
        'clear': 1,
        'snow': 2,
        'light_fog': 3,
        'rain': 4,
        'unclear': 5
    }
    return label_mapping.get(label, None)


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
                 transforms=None, label_file: str = ""):
        """
        :param split: Which split to load (train, val, test)
        :param normalize: Whether to normalize/standardize images before returning them in __getitem__()
        :param label_file: Path to the JSON file containing image labels
        """
        self.transforms = transforms
        self.normalize = normalize
        self.labels = self.load_labels(label_file)

        # Load file paths
        _, file_paths = get_base_path(split, base_path=base_path, assume_existing_files_only=True)
        file_paths = sorted(file_paths)

        # Filter out files with invalid labels
        self.img_filepaths = [path for path in file_paths if self.labels.get(path.name.split('.')[0], -1) != -1]

    def load_labels(self, label_file):
        with open(label_file, 'r') as f:
            labels = json.load(f)
        return labels

    def __getitem__(self, index):
        img_path = self.img_filepaths[index]
        try:
            img = PIL.Image.open(img_path)
            #img_tensor = torchvision.transforms.ToTensor()(img)
            if self.transforms:
                img_tensor = self.transforms(img)
            if self.normalize:
                normalize = torchvision.transforms.Normalize(mean=[0.364, 0.349, 0.331],
                                                             std=[0.183, 0.183, 0.184])
                img_tensor = normalize(img_tensor)

            # Extract filename without extension to match labels
            filename = img_path.name.split('.')[0]
            label = self.labels.get(filename, "unknown")  # Default label if not found
            return img_tensor, label

        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")
            return torch.zeros(3, 224, 224), -1


    def __len__(self):
        return len(self.img_filepaths)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = SeeingThroughFogImageDataset('train', label_file=r'C:\Users\venkatesh\Desktop\Personal '
                                                               r'Projects\weather_recognition\resources'
                                                               r'\dense_weather_labels.json', normalize=True, transforms=transforms.Compose(
                                                    [transforms.Resize((224, 224)), transforms.ToTensor()]))
    img_tensor, label = dataset[0]
    # Convert the tensor to a NumPy array and transpose the dimensions for plotting
    # The tensor is in CxHxW format, but matplotlib expects HxWxC format
    img_array = img_tensor.numpy().transpose(1, 2, 0)
    img_array = np.clip(img_array, 0, 1)
    plt.imshow(img_array)
    plt.title(label)  # Optionally display the label as the title
    plt.show()
