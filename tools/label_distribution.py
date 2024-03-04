import yaml
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Replace these paths with the actual paths to your files
split_file_path = r'C:\Users\venkatesh\Desktop\Personal Projects\weather_recognition\resources\dense_images_split.yaml'
label_file_path = r'C:\Users\venkatesh\Desktop\Personal Projects\weather_recognition\resources\dense_weather_labels.json'
output_dir = Path(r'C:\Users\venkatesh\Desktop\Personal Projects\weather_recognition\resources')

# Load dataset splits
with open(split_file_path, 'r') as file:
    splits = yaml.safe_load(file)

# Load labels
with open(label_file_path, 'r') as file:
    labels = json.load(file)

# Prepare the plot
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)
splits_names = ['train', 'val', 'test']

for ax, split_name in zip(axs, splits_names):
    # Extract identifiers from file paths and filter labels for the current split
    split_identifiers = [Path(p).stem for p in splits[split_name]]  # Extract the filename without extension
    split_labels = {k: v for k, v in labels.items() if k in split_identifiers}

    if not split_labels:
        print(f"No matching labels found for {split_name} split.")
        continue

    # Count occurrences of each weather condition
    label_counts = pd.Series(split_labels).value_counts().sort_index()

    # Plot
    if not label_counts.empty:
        label_counts.plot(kind='bar', ax=ax)
        ax.set_title(f'{split_name.capitalize()} Split')
        ax.set_xlabel('Weather Condition')
        ax.set_ylabel('Frequency')
        for container in ax.containers:
            ax.bar_label(container, label_type='edge')
    else:
        print(f"No data to plot for {split_name} split")

plt.suptitle('Weather Condition Distribution Across Splits')
plt.tight_layout()
plt.savefig(output_dir / 'weather_condition_distribution.png')
plt.show()
