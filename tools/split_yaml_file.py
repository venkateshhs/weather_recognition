import yaml
import json
from pathlib import Path

# File paths
images_split_file = r'C:\Users\venkatesh\Desktop\Personal Projects\weather_recognition\resources\dense_images_split' \
                    r'.yaml'
labels_file = r'C:\Users\venkatesh\Desktop\Personal Projects\weather_recognition\resources\dense_weather_labels.json'
output_file = r'C:\Users\venkatesh\Desktop\Personal Projects\weather_recognition\resources\filtered_labels_split.yaml'

with open(images_split_file, 'r') as file:
    images_split = yaml.safe_load(file)


with open(labels_file, 'r') as file:
    weather_labels = json.load(file)


def extract_id(image_path):
    return Path(image_path).stem


filtered_split = {'train': [], 'test': [], 'val': []}
for split in ['train', 'test', 'val']:
    for image_path in images_split[split]:
        image_id = extract_id(image_path)
        print(image_id)
        weather_label = weather_labels.get(image_id)
        if weather_label in ['clear', 'snow']:
            filtered_split[split].append(image_path)

with open(output_file, 'w') as file:
    yaml.dump(filtered_split, file, default_flow_style=False)

print("Filtered YAML file has been created.")
