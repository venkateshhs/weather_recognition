import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from provider.see_through_fog_images import SeeingThroughFogImageDataset
from util.utils import label_to_int
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

log_folder = r'C:\Users\venkatesh\Desktop\Personal Projects\weather_recognition\logs'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_folder, f'training_log_{current_time}.log')

logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config['training']


def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))

    logging.info(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    return conf_matrix


def visualize_confusion_matrix(conf_matrix, num_classes):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def train_model(dataset, config, val_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Starting training with learning rate {config['learning_rate']}, batch size: {config['batch_size']} and {config['epochs']} epochs")

    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{config["epochs"]}', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        logging.info(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {running_loss / len(dataloader)}")
    logging.info(f'Evaluating training set')
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    train_conf_matrix = evaluate_model(model, train_loader, device, num_classes=3)
    logging.info(f'training set confusion metrix {train_conf_matrix}')

    logging.info("Evaluating on Validation Set")
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    val_conf_matrix = evaluate_model(model, val_loader, device, num_classes=3)
    logging.info(f'Validation set confusion metrix {val_conf_matrix}')

    logging.info("Evaluating on test Set")
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    test_conf_matrix = evaluate_model(model, test_loader, device, num_classes=3)
    logging.info(f'Validation set confusion metrix {test_conf_matrix}')


if __name__ == "__main__":
    config = load_config(r'C:\Users\venkatesh\Desktop\Personal Projects\weather_recognition\conf\config.yaml')

    # Adjust paths and parameters as necessary
    dataset = SeeingThroughFogImageDataset('train', label_file=r'C:\Users\venkatesh\Desktop\Personal '
                                                               r'Projects\weather_recognition\resources'
                                                               r'\dense_weather_labels.json', normalize=True,
                                           transforms=transforms.Compose(
                                               [transforms.Resize((224, 224)), transforms.ToTensor()]))
    val_dataset = SeeingThroughFogImageDataset('val', label_file=r'C:\Users\venkatesh\Desktop\Personal '
                                                                 r'Projects\weather_recognition\resources'
                                                                 r'\dense_weather_labels.json', normalize=True,
                                               transforms=transforms.Compose(
                                                   [transforms.Resize((224, 224)), transforms.ToTensor()]))
    test_dataset = SeeingThroughFogImageDataset('test', label_file=r'C:\Users\venkatesh\Desktop\Personal '
                                                                   r'Projects\weather_recognition\resources'
                                                                   r'\dense_weather_labels.json', normalize=True,
                                                transforms=transforms.Compose(
                                                    [transforms.Resize((224, 224)), transforms.ToTensor()]))

    # Apply label transformation
    for ds in [dataset, val_dataset, test_dataset]:
        ds.labels = {k: label_to_int(v) for k, v in ds.labels.items() if label_to_int(v) != -1}
    train_model(test_dataset, config, val_dataset, dataset)
