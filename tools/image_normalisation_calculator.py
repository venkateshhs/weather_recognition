import numpy as np
import torch.utils.data
import tqdm

from provider.see_through_fog_images import SeeingThroughFogImageDataset


def main():
    train_dataset = SeeingThroughFogImageDataset("", split="train", normalize=False)
    train_dataset.initialize_in_worker(0)
    channel_values_sum = torch.zeros(3)
    channel_squared_values_sum = torch.zeros(3)
    pixel_count = 0
    samples_progress_bar = tqdm.tqdm(np.random.permutation(range(len(train_dataset))))
    for sample_idx in samples_progress_bar:
        sample = train_dataset[sample_idx]
        img = sample.data_instance.as_tensor()
        # dimensions should be (C x H x W)
        channel_values_sum += torch.sum(img, dim=[1, 2])
        channel_squared_values_sum += torch.sum(img ** 2, dim=[1, 2])
        pixel_count += img.shape[1] * img.shape[2]

    channel_means = channel_values_sum / pixel_count
    channel_stds = torch.sqrt(1 / pixel_count * channel_squared_values_sum - channel_means ** 2)

    # Convert tensors to floats for formatting
    mean_vals = channel_means.tolist()  # Convert to list of floats
    std_vals = channel_stds.tolist()  # Convert to list of floats
    progress_desc = f"<µ={mean_vals[0]:.3f}, {mean_vals[1]:.3f}, {mean_vals[2]:.3f}, s={std_vals[0]:.3f}, {std_vals[1]:.3f}, {std_vals[2]:.3f}> "
    samples_progress_bar.set_description(progress_desc)
    print(progress_desc)
