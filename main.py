# Installing the requirements
print('Installing Requirements... ',end='')
!pip install lightning
print('Done')

# Importing Libraries
print('Importing Libraries... ',end='')
import os
from pathlib import Path
import pandas as pd
import torchaudio
import zipfile
from torchaudio.transforms import Resample
import IPython.display as ipd
from matplotlib import pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch
print('Done')

from google.colab import drive
drive.mount('/content/drive')

# Extract data
with zipfile.ZipFile("/content/drive/MyDrive/M.Tech. Sem 2/Deep Learning/Assignment 2/archive.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/")

# Loading dataset
path = Path('/content')
df = pd.read_csv('/content/meta/esc50.csv')
print("First 5 samples: \n", df.head())

categories = sorted(df["category"].unique())
categories

"""ESC-10 (Environmental Sound Classification - 10) is a dataset designed for the task of audio classification. It is a subset of the larger ESC-50 dataset, which consists of 50 classes of environmental sounds. The ESC-10 dataset, as the name suggests, focuses on a smaller set of 10 classes."""

print("Total values of esc10: \n", df['esc10'].value_counts())

"""### The shape of the waveform is represented as `torch.Size([1, 220500])`.

* The first dimension (1): This dimension usually represents the number of channels in the audio. In this case, it's 1, indicating a mono audio signal. For stereo audio, this would typically have two channels.

* The second dimension (220500): This dimension represents the length of the audio signal in samples. In this example, the audio waveform has 220,500 samples.
"""

# Getting list of raw audio files
wavs = list(path.glob('audio/*'))  # List all audio files in the 'audio' directory using pathlib.Path.glob

# Visualizing data
waveform, sample_rate = torchaudio.load(wavs[0])  # Load the waveform and sample rate of the first audio file using torchaudio

print("Shape of waveform: {}".format(waveform.size()))  # Print the shape of the waveform tensor
print("Sample rate of waveform: {}".format(sample_rate))  # Print the sample rate of the audio file

# Display the audio using IPython.display.Audio
ipd.Audio(waveform, rate=sample_rate)  # Create an interactive audio player for the loaded waveform

"""Plot the waveform of the audio signal"""

# Calculate time values for x-axis
time = torch.arange(0, waveform.size(1)) / sample_rate

# Plotting the waveform using plt.plot
plt.figure(figsize=(10, 4))
plt.plot(time.numpy(), waveform.t().numpy(), color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')
# Display the plot
plt.show()

"""### Classes to create `Dataset` and `Dataloader`"""

class CustomDataset(Dataset):
    def __init__(self, dataset, **kwargs):
        # Initialize CustomDataset object with relevant parameters
        # dataset: "train", "val", or "test"
        # kwargs: Additional parameters like data directory, dataframe, folds, etc.

        # Extract parameters from kwargs
        self.data_directory = kwargs["data_directory"]
        self.data_frame = kwargs["data_frame"]
        self.validation_fold = kwargs["validation_fold"]
        self.testing_fold = kwargs["testing_fold"]
        self.esc_10_flag = kwargs["esc_10_flag"]
        self.file_column = kwargs["file_column"]
        self.label_column = kwargs["label_column"]
        self.sampling_rate = kwargs["sampling_rate"]
        self.new_sampling_rate = kwargs["new_sampling_rate"]
        self.sample_length_seconds = kwargs["sample_length_seconds"]

        """
          We are only working with esc10 dataset so filter the dataframe based on that
        """
        if self.esc_10_flag:
            self.data_frame = self.data_frame.loc[self.data_frame['esc10'] == True]


        if dataset == "train":
            self.data_frame = self.data_frame.loc[
                (self.data_frame['fold'] != self.validation_fold) & (self.data_frame['fold'] != self.testing_fold)]
        elif dataset == "val":
            self.data_frame = self.data_frame.loc[self.data_frame['fold'] == self.validation_fold]
        elif dataset == "test":
            self.data_frame = self.data_frame.loc[self.data_frame['fold'] == self.testing_fold]

        # Get unique categories from the filtered dataframe
        self.categories = sorted(self.data_frame[self.label_column].unique()) # In this label_column = 'category'

        # Initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []

        # Initialize dictionaries for category-to-index and index-to-category mapping
        self.category_to_index = {}
        self.index_to_category = {}

        for i, category in enumerate(self.categories):
            self.category_to_index[category] = i
            self.index_to_category[i] = category

        # Populate file names and labels lists by iterating through the dataframe
        for ind in tqdm(range(len(self.data_frame))):
            row = self.data_frame.iloc[ind]
            file_path = self.data_directory / "audio" / row[self.file_column]
            self.file_names.append(file_path)
            self.labels.append(self.category_to_index[row[self.label_column]])

        self.resampler = torchaudio.transforms.Resample(self.sampling_rate, self.new_sampling_rate)

        # Window size for rolling window sample splits (unfold method)
        if self.sample_length_seconds == 2:
            self.window_size = self.new_sampling_rate * 2
            self.step_size = int(self.new_sampling_rate * 0.75)
        else:
            self.window_size = self.new_sampling_rate
            self.step_size = int(self.new_sampling_rate * 0.5)

    """
      audio_tensor.unfold(1, self.window_size, self.step_size):
        --- This method is used to create overlapping splits of the audio data.
        --- It unfolds the tensor along dimension 1 (time dimension) with a window size of self.window_size and a step size of self.step_size.

      Example: If self.window_size is 16000 (representing 1 second of audio at a sampling rate of 16000 Hz) and self.step_size is 12000,
              it means you are creating overlapping windows with a size of 16000 samples and moving forward by 12000 samples for each window.

      Let's say your original audio tensor had shape (1, num_samples), the unfolded tensor might have a shape like (1, num_windows, self.window_size).

    """

    def __getitem__(self, index):
        # Split audio files with overlap, pass as stacked tensors tensor with a single label
        path = self.file_names[index]
        audio_file = torchaudio.load(path, format=None, normalize=True)
        audio_tensor = self.resampler(audio_file[0])
        # print("Shape of Audio Tensor: ", audio_tensor.shape)
        splits = audio_tensor.unfold(1, self.window_size, self.step_size)
        samples = splits.permute(1, 0, 2)
        # print("Samples Shape: ", samples.shape)
        # print("Window Size: ", self.window_size)
        # samples = samples.unsqueeze(0)
        return samples, self.labels[index]

    def __len__(self):
        return len(self.file_names)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        # Initialize the CustomDataModule with batch size, number of workers, and other parameters
        super().__init__()
        self.batch_size = kwargs["batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.data_module_kwargs = kwargs

    def setup(self, stage=None):
        # Define datasets for training, validation, and testing during Lightning setup

        # If in 'fit' or None stage, create training and validation datasets
        if stage == 'fit' or stage is None:
            self.training_dataset = CustomDataset(dataset="train", **self.data_module_kwargs)
            self.validation_dataset = CustomDataset(dataset="val", **self.data_module_kwargs)

        # If in 'test' or None stage, create testing dataset
        if stage == 'test' or stage is None:
            self.testing_dataset = CustomDataset(dataset="test", **self.data_module_kwargs)

    def train_dataloader(self):
        # Return DataLoader for training dataset
        return DataLoader(self.training_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collate_function,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        # Return DataLoader for validation dataset
        return DataLoader(self.validation_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=self.collate_function,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        # Return DataLoader for testing dataset
        return DataLoader(self.testing_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=self.collate_function,
                          num_workers=self.num_workers)

    def collate_function(self, data):
        """
        Collate function to process a batch of examples and labels.

        Args:
            data: a tuple of 2 tuples with (example, label) where
                example are the split 1 second sub-frame audio tensors per file
                label = the label

        Returns:
            A list containing examples (concatenated tensors) and labels (flattened tensor).
        """
        examples, labels = zip(*data)
        examples = torch.stack(examples)
        examples = examples.reshape(examples.size(0), 1, -1)

        labels = torch.flatten(torch.tensor(labels))

        return [examples, labels]

"""# Model Architectures

Importing libaries
"""

!pip install wandb

import torch
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import wandb
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_optimizer(name, model, lr):
  if name == "adam":
    optimizer = optim.Adam(model.parameters(), lr=lr)

  elif name == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)

  else:
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

  return optimizer

class SoundClassifierTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, num_classes, lr=0.001, using_wandb=False, early_stop=True, optimizer="adam"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(optimizer, model, lr)
        # self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.training_losses = []
        self.accuracies = []
        self.num_classes = num_classes
        self.val_losses = []
        self.val_accuracies = []
        self.l2_penalty = 1e-5
        self.using_wandb = using_wandb
        self.early_stop = early_stop

    def train(self, epochs):
        best_val_accuracy = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()  # Training Mode of torch model
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                self.model = self.model.to(device)
                outputs, blank = self.model(inputs)  # Forward pass
                loss = self.criterion(outputs, labels)

                loss.backward()  # Backward pass
                self.optimizer.step()  # Update the weights

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Average training loss and accuracy for this epoch
            avg_loss = total_loss / len(self.train_loader)
            # Save training loss for plotting
            self.training_losses.append(avg_loss)

            accuracy = correct / total
            self.accuracies.append(accuracy * 100)

            # Validation
            val_loss, val_accuracy = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy * 100)

            # Early Stopping
            if self.early_stop:
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation accuracy.")
                        break

            # log metrics to wandb
            if self.using_wandb:
              wandb.log({"acc": accuracy, "loss": avg_loss, 'val_acc': val_accuracy, 'val_loss': val_loss})

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")

        # Plot the loss-epoch curve
        # self.plot_loss_and_accuracy_curves()
        # Plot the loss-epoch curve
        # self.plot_loss_and_accuracy_curves()
        return {'val_accuracy': best_val_accuracy}

    def validate(self):
        self.model.eval()  # Evaluation Mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                self.model = self.model.to(device)
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = correct / total

        return avg_val_loss, val_accuracy

    def evaluate(self):
        self.model.eval()  # Evaluation Mode
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                self.model = self.model.to(device)
                outputs, _ = self.model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                predicted = predicted.to(device)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        accuracy = correct / total
        test_f1 = f1_score(all_labels, all_predictions, average='weighted')
        return {'accuracy': accuracy, 'f1_score': test_f1, 'labels': all_labels, 'predictions': all_predictions, 'probabilities': all_probabilities}

    def plot_loss_and_accuracy_curves(self):
        epochs = range(1, len(self.training_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot Loss
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs, self.training_losses, label='Training Loss', color=color)
        ax1.plot(epochs, self.val_losses, label='Validation Loss', linestyle='dashed', color='orange')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        # Plot Accuracy
        color = 'tab:blue'
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(epochs, self.accuracies, label='Training Accuracy', color=color)
        ax2.plot(epochs, self.val_accuracies, label='Validation Accuracy', linestyle='dashed', color='green')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper left')

        fig.tight_layout()
        plt.suptitle('Loss and Accuracy vs. Epoch')
        plt.show()


    def plot_confusion_matrix(self, labels, predictions, title):
        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        print("Confusion Matrix")
        print(cm)
        print("\n\n\n")

        class_names = [str(i) for i in range(self.num_classes)]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def plot_auc_roc(self, labels, probabilities, n_classes, title):
        # Calculate FPR, TPR, and AUC for each class
        fprs = []
        tprs = []
        aucs = []
        for i in range(n_classes):  # Iterate through classes 0 to 9
            fpr, tpr, _ = roc_curve(labels, [prob[i] for prob in probabilities], pos_label=i)
            roc_auc = auc(fpr, tpr)
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(roc_auc)

        # Plot the ROC curves for each class
        plt.figure(figsize=(8, 6))  # Adjust figure size as needed
        for i in range(n_classes):
            plt.plot(fprs[i], tprs[i], label=f"Class {i} (AUC = {aucs[i]:.2f})")

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Each Class')
        plt.legend(loc="lower right")
        plt.show()

    def report_parameters(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

        print(f"Total Trainable Parameters: {trainable_params}")
        print(f"Total Non-trainable Parameters: {non_trainable_params}")

"""# Architecture-1: CNN"""

class CNNSoundClassification(nn.Module):
    def __init__(self, num_classes):
        super(CNNSoundClassification, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3, stride=1)
        self.elu1 = nn.ELU()
        self.maxpool1 = nn.MaxPool1d(2) # 1x72000x16

        self.dropout = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.elu2 = nn.ELU()
        self.maxpool2 = nn.MaxPool1d(2) # 1x36000x32

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.elu3 = nn.ELU()
        self.maxpool3 = nn.MaxPool1d(4) # 1x9000x64

        self.conv4 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.elu4 = nn.ELU()
        self.maxpool4 = nn.MaxPool1d(4) # 1x2250x32

        self.conv5 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)
        self.elu5 = nn.ELU()
        self.maxpool5 = nn.MaxPool1d(2, 2) # 1x1125x16

        # Output Layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16 * 1125, num_classes)

    def forward(self, x):
        # Apply convolutional layers with activation and pooling
        x = x.to(device)
        x = self.maxpool1(self.elu1(self.conv1(x)))
        x = self.maxpool2(self.elu2(self.conv2(x)))

        x = self.dropout(x)

        x = self.maxpool3(self.elu3(self.conv3(x)))
        x = self.maxpool4(self.elu4(self.conv4(x)))

        x = self.dropout(x)

        x = self.maxpool5(self.elu5(self.conv5(x)))

        # Features
        features = x

        # Flatten and apply linear layer
        x = self.flatten(x)
        x = self.linear(x)

        # Apply softmax activation
        x = F.softmax(x, dim=1)

        return x, features

"""### Get the dataset"""

def get_the_dataset(batch_size, lr, valid_samp, test_samp=1, num_workers=2):
    esc_data = CustomDataModule(batch_size=batch_size,
                                          num_workers=num_workers,
                                          data_directory=path,
                                          data_frame=df,
                                          validation_fold=valid_samp,
                                          testing_fold=test_samp,  # set to 0 for no test set
                                          esc_10_flag=True,
                                          file_column='filename',
                                          label_column='category',
                                          sampling_rate=44100,
                                          new_sampling_rate=16000,  # new sample rate for input
                                          sample_length_seconds=1  # new length of input in seconds
                                          )

    esc_data.setup()

    return esc_data

"""### wandB Logs"""

def init_wandb(project, name, config):
  # start a new wandb run to track this script
  wandb.init(
      # set the wandb project where this run will be logged
      project=project,
      name=name,
      # track hyperparameters and run metadata
      config=config
  )

"""Plot hyperparameter tuning results"""

def plot_htuning(accuracies, batch_sizes, learning_rates, ffn_hiddens):
    # Creating combinations
    combinations = [(batch_size, lr, ffn_hidden) for batch_size in batch_sizes for lr in learning_rates for ffn_hidden in ffn_hiddens]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))  # Increase the width of the plot

    # Creating color map for each combination
    colors = plt.cm.viridis(np.linspace(0, 1, len(combinations)))

    # Plotting bars
    bars = ax.bar(range(len(combinations)), accuracies, color=colors)

    # Adding labels and title
    ax.set_xticks(range(len(combinations)))
    # ax.set_xticklabels([f'{c[0]}, {c[1]}, {c[2]}' for c in combinations], rotation=45, ha='right', fontsize=10)  # Adjust rotation and fontsize
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy for Hyperparameter Combinations')

    # Adding legend outside the plot
    ax.legend(bars, [f'{bs}, {lr}, {optim}' for bs, lr, optim in combinations], loc='upper left', bbox_to_anchor=(1, 1))

    # Adding grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adding some extra space at the bottom for better visibility of x-axis labels
    plt.subplots_adjust(bottom=0.2)

    # Customize borders and spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    # Show the plot
    plt.show()

"""### Training with Hyperparameter Tuning"""

# Track best hyperparams
best_hyperparams = {
    'acc' : 0,
    'hparams' : [],
    'optimizer': "adam"
}

epochs = 100
num_classes = 10
test_samp = 1 # """ Do not change this!! """

batch_sizes = [16, 32]
learning_rates = [0.001, 0.01]
optimizers = ["sgd", "adam"]
avg_acc_list = []

# Iterate through combinations
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for optimizer in optimizers:
          avg_acc = 0
          for val_fold in range(2, 6):
            print(f"Evaluating hyperparameters:\nBatch Size: {batch_size}\nLearning Rate: {learning_rate}\nVal Fold:{val_fold}\nOptimizer: {optimizer}\n\n")
            valid_samp = val_fold # Use any value ranging from 2 to 5 for k-fold validation (valid_fold)

            # Get the dataset
            esc_data = get_the_dataset(batch_size, learning_rate, valid_samp, test_samp)
            # Initializing Model Object
            model = CNNSoundClassification(num_classes)
            model = model.to(device)
            # Initializing Model Trainer
            model_trainer = SoundClassifierTrainer(model,
                                                   esc_data.train_dataloader(),
                                                   esc_data.val_dataloader(),
                                                   esc_data.test_dataloader(),
                                                   num_classes,
                                                   learning_rate,
                                                   optimizer=optimizer)

            print(f"Results of model with Batch Size: {batch_size} Learning Rate: {learning_rate} \n")
            # Training
            outputs = model_trainer.train(epochs)
            # Evaluation
            eval_outputs = model_trainer.evaluate()

            print(f"F1-score: {eval_outputs['f1_score']:.2f}")
            print(f"Accuracy: {eval_outputs['accuracy']* 100:.2f}%")

            # Calculate avg accuracy
            avg_acc += outputs['val_accuracy']

            model_trainer.plot_confusion_matrix(eval_outputs['labels'], eval_outputs['predictions'], "Confusion Matrix for CNN Architecture")

            model_trainer.plot_auc_roc(eval_outputs['labels'], eval_outputs['probabilities'], num_classes, "AUC ROC for CNN Architecture")

            print("\n\n")

          # Add to Best Hyperparams
          avg_acc = avg_acc / 4

          if avg_acc > best_hyperparams['acc']:
            best_hyperparams['acc'] = avg_acc
            best_hyperparams['optimizer'] = optimizer
            best_hyperparams['hparams'] = [batch_size, learning_rate]

          avg_acc_list.append(avg_acc)

print("Best Hyperparams: ", best_hyperparams)

plot_htuning(avg_acc_list, batch_sizes, learning_rates, optimizers)

"""### Training with Best Hyperparams"""

wandb.login(key='cd501af2a321aac1f8cbddd1189b43863a2f2874')

training_params = {
  "test_samp": 1,
  "valid_samp": 3,
  "learning_rate": best_hyperparams['hparams'][1] if len(best_hyperparams['hparams']) != 0 else 0.001,
  "batch_size": best_hyperparams['hparams'][0] if len(best_hyperparams['hparams']) != 0 else 32,
  "optimizer": best_hyperparams['optimizer'],
  "architecture": "CNN",
  "dataset": "esc10",
  "epochs": 100,
  "num_classes": 10
}

print(f"Training:\nBatch Size: {training_params['batch_size']}\nLearning Rate: {training_params['learning_rate']}\nOptimizer: {training_params['optimizer']}\n")
# Get the dataset
esc_data = get_the_dataset(training_params['batch_size'],
                           training_params['learning_rate'],
                           training_params['valid_samp'],
                           training_params['test_samp']
                        )

# Initialize wandb
init_wandb(
  project = 'Architecture-1',
  name = f"LR:{training_params['learning_rate']} BS:{training_params['batch_size']}",
  config = training_params
)

# Initializing Model Object
model = CNNSoundClassification(training_params['num_classes'])
model = model.to(device)
# Initializing Model Trainer
model_trainer = SoundClassifierTrainer(model,
                                       esc_data.train_dataloader(),
                                       esc_data.val_dataloader(),
                                       esc_data.test_dataloader(),
                                       training_params['num_classes'],
                                       training_params['learning_rate'],
                                       using_wandb=True,
                                       early_stop=False,
                                       optimizer=training_params['optimizer'])

print(f"Results of model with Batch Size: {training_params['batch_size']} Learning Rate: {training_params['learning_rate']} \n")
# Training
outputs = model_trainer.train(training_params['epochs'])
print("\n\n")
wandb.finish()

"""Evaluating the model performance"""

# Print and log test set metrics
# Evaluation
eval_outputs = model_trainer.evaluate()

print(f"F1-score: {eval_outputs['f1_score']:.2f}")
print(f"Accuracy: {eval_outputs['accuracy']* 100:.2f}%")

"""Plot Confusion Matrix"""

model_trainer.plot_confusion_matrix(eval_outputs['labels'], eval_outputs['predictions'], "Confusion Matrix for CNN Architecture")

"""Plot AUC-ROC curve"""

model_trainer.plot_auc_roc(eval_outputs['labels'], eval_outputs['probabilities'], training_params['num_classes'], "AUC ROC for CNN Architecture")

model_trainer.plot_loss_and_accuracy_curves()

"""Report Parameters"""

model_trainer.report_parameters()

"""# Architecture-2: Transformer Encoder

Importing necessary libraries
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Transformer Encoder Architecture"""

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, n_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.h_dim = model_dim // n_heads

        # Define the linear layers for query, key, and value
        self.query_layer = nn.Linear(model_dim, model_dim)
        self.key_layer = nn.Linear(model_dim, model_dim)
        self.value_layer = nn.Linear(model_dim, model_dim)

        # Output linear layer
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        x = x.to(device)
        batch_size, seq_len, model_dim = x.size()

        # Linear transformations for query, key, and value
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.h_dim).transpose(1, 2)

        # Scaled dot-product attention
        values, attention = self.scaled_dot(q, k, v)

        # Reshape and linear transformation for the output
        values = values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.h_dim)
        output = self.linear(values)

        return output

    def scaled_dot(self, q, k, v):
        dk = q.size(-1)
        dotp = torch.matmul(q, k.transpose(-2, -1))
        scaled = dotp / math.sqrt(dk)
        attention = F.softmax(scaled, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

# Class for LayerNormalization
class LayerNormalization(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = x.to(device)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.shift

# Class for Fully Connected Layer
class FullyConnectedLayer(nn.Module):
  def __init__(self, model_dim, num_classes):
    super().__init__()
    self.network = nn.Sequential(
        nn.Linear(model_dim, num_classes),
        nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = x.to(device)
    return self.network(x)

# Class for Fully Connected Layer
class FFNetwork(nn.Module):
  def __init__(self, model_dim, h_layers):
    super().__init__()
    self.network = nn.Sequential(
        nn.Linear(model_dim, h_layers),
        nn.ReLU(),
        nn.Linear(h_layers, model_dim),
        nn.ReLU()
    )

  def forward(self, x):
    x = x.to(device)
    return self.network(x)

# Class for Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
  def __init__(self, model_dim, fc_layers, n_heads, num_classes, num_att_blocks):
    super(TransformerEncoderLayer, self).__init__()
    self.attentions = nn.ModuleList([MultiHeadSelfAttention(model_dim=model_dim, n_heads=n_heads) for _ in range(num_att_blocks)])
    self.norms = nn.ModuleList([LayerNormalization(dim=[model_dim]) for _ in range(num_att_blocks)])
    self.ffn = FFNetwork(model_dim=model_dim, h_layers=fc_layers)
    self.norm3 = LayerNormalization(dim=[model_dim])

  def forward(self, x):
    x = x.to(device)

    # Multi-head self-attention and normalization blocks
    for attention, norm in zip(self.attentions, self.norms):
        x = x + attention(x)
        x = norm(x)

    # FFNetwork and final normalization block
    x = x + self.ffn(x)
    x = self.norm3(x)
    return x

# Class for Transformer Encoder Architecture Network
class TransformerEncoder(nn.Module):
  def __init__(self, model_dim, te_layers, fc_layers, n_heads, num_classes, num_att_blocks):
    super().__init__()
    self.layers = nn.ModuleList([TransformerEncoderLayer(model_dim, fc_layers, n_heads, num_classes, num_att_blocks) for _ in range(te_layers)])

  def forward(self, x):
    for layer in self.layers:
        x = x.to(device)
        x = layer(x)
    return x, []

"""Combined Model Class (CNN + Transformer)"""

class CNNTEModel(nn.Module):
    def __init__(self, cnn_model, transformer_encoder, model_dim=8, num_classes=10, max_len=1125):
        super(CNNTEModel, self).__init__()
        self.cnn_model = cnn_model
        self.model_dim = model_dim
        self.max_len = max_len
        self.transformer_encoder = transformer_encoder
        self.fc = FullyConnectedLayer(model_dim=model_dim, num_classes=num_classes)

    def get_positional_encoding(self, x):
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.model_dim, 2, dtype=torch.float) * -(math.log(10000.0) / self.model_dim))

        # Calculate positional encoding using broadcasting
        pe = torch.zeros(1, self.max_len, self.model_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        pe = pe.repeat(x.size(0), 1, 1)

        return pe

    def get_cls_token(self, x):
        cls_token = nn.Parameter(torch.randn(1, 1, self.model_dim))
        cls_token = cls_token.repeat(x.size(0), 1, 1)

        return cls_token

    def forward(self, x):
        x = x.to(device)
        # Set CNN model to device
        self.cnn_model = self.cnn_model.to(device)
        # Get CNN features to feed into TE model
        _, cnn_features = self.cnn_model(x)  # Pass input through CNN

        cnn_features = cnn_features.permute(0, 2, 1) # Change (32, 16, 4500) ---> (32, 4500, 16)

        # Set both
        cnn_features = cnn_features.to(device)

        pos_en = self.get_positional_encoding(cnn_features)
        pos_en = pos_en.to(device)

        enc_features = pos_en + cnn_features
        enc_features = enc_features.to(device)

        cls_token = self.get_cls_token(cnn_features)
        cls_token = cls_token.to(device)

        tokenized_features = torch.cat([cls_token, enc_features], dim=1)

        self.transformer_encoder = self.transformer_encoder.to(device)
        output, _  = self.transformer_encoder(tokenized_features)

        # Classification head
        cls_token_embedding = output[:, 0, :]  # Extract <cls> token embedding
        logits = self.fc(cls_token_embedding)  # Pass to classification head

        return logits, output

"""Training with Hyperparameter Tuning"""

epochs = 100
num_classes = 10

batch_sizes = [16, 32]
learning_rates = [0.001, 0.01]
ffn_hiddens = [64, 128]

# Data Setup
test_samp = 1 # """ Do not change this!! """
num_workers = 2 # Free to change

model_dim = 16
# ffn_hidden = 64
num_layers = 1
num_head = 2
num_att_blocks = 4

best_hyperparams = {
    'acc' : 0,
    'hparams' : [],
    'optimizer': "adam",
    'ffn_hidden': 64
}

avg_acc_list = []

# Iterate through combinations
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for ffn_hidden in ffn_hiddens:
          avg_acc = 0
          for val_fold in range(2, 6):
            print(f"Evaluating hyperparameters:\nBatch Size: {batch_size}\nLearning Rate: {learning_rate}\nHidden Layers:{ffn_hidden}")
            valid_samp = val_fold # Use any value ranging from 2 to 5 for k-fold validation (valid_fold)

            esc_data = CustomDataModule(batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  data_directory=path,
                                                  data_frame=df,
                                                  validation_fold=valid_samp,
                                                  testing_fold=test_samp,  # set to 0 for no test set
                                                  esc_10_flag=True,
                                                  file_column='filename',
                                                  label_column='category',
                                                  sampling_rate=44100,
                                                  new_sampling_rate=16000,  # new sample rate for input
                                                  sample_length_seconds=1  # new length of input in seconds
                                                  )

            esc_data.setup()

            cnn_model = CNNSoundClassification(num_classes)
            te_model = TransformerEncoder(model_dim, num_layers, ffn_hidden, num_head, num_classes, num_att_blocks)
            # pos_encoder = PositionalEncoding(model_dim)

            cnnte_model = CNNTEModel(cnn_model, te_model, model_dim=model_dim)

            # Initializing Model Trainer
            model_trainer = SoundClassifierTrainer(cnnte_model,
                                                  esc_data.train_dataloader(),
                                                  esc_data.val_dataloader(),
                                                  esc_data.test_dataloader(),
                                                  num_classes,
                                                  learning_rate)

            print(f"Results of model with Batch Size: {batch_size} Learning Rate: {learning_rate} \n")
            # Training
            outputs = model_trainer.train(epochs)
            # Evaluation
            eval_outputs = model_trainer.evaluate()
            print("\n\n")
            print(f"F1-score: {eval_outputs['f1_score']:.2f}")
            print(f"Accuracy: {eval_outputs['accuracy']* 100:.2f}%")

            model_trainer.plot_confusion_matrix(eval_outputs['labels'], eval_outputs['predictions'], f"Confusion Matrix ")

            model_trainer.plot_auc_roc(eval_outputs['labels'], eval_outputs['probabilities'], num_classes, f"AUC ROC ")

            model_trainer.plot_loss_and_accuracy_curves()

            # Calculate avg accuracy
            avg_acc += outputs['val_accuracy']

            print("\n\n")

          # Add to Best Hyperparams
          avg_acc = avg_acc / 4

          if avg_acc > best_hyperparams['acc']:
            best_hyperparams['acc'] = avg_acc
            best_hyperparams['hparams'] = [batch_size, learning_rate]
            best_hyperparams['ffn_hidden'] = ffn_hidden

          avg_acc_list.append(avg_acc)

print("Best Hyperparams: ", best_hyperparams)

plot_htuning(avg_acc_list, batch_sizes, learning_rates, ffn_hiddens)

"""Training with Best Hyperparameters"""

training_params = {
  "test_samp": 1,
  "valid_samp": 3,
  "learning_rate": best_hyperparams['hparams'][1] if len(best_hyperparams['hparams']) != 0 else 0.001,
  "batch_size": best_hyperparams['hparams'][0] if len(best_hyperparams['hparams']) != 0 else 32,
  "optimizer": best_hyperparams['optimizer'],
  "architecture": "CNN + Transformer",
  "dataset": "esc10",
  "epochs": 100,
  "num_classes": 10,
  "model_dim": 16,
  "ffn_hidden": best_hyperparams['ffn_hidden'],
  "num_layers": 1,
  "num_att_blocks": 4
}

# Training for each head
for num_head in [1, 2, 4]:
    print(f"Training:\nBatch Size: {training_params['batch_size']}\nLearning Rate: {training_params['learning_rate']}\nHead: {num_head}")

    # Get the dataset
    esc_data = get_the_dataset(training_params['batch_size'],
                              training_params['learning_rate'],
                              training_params['valid_samp'],
                              training_params['test_samp']
                            )

    # Initialize wandb
    init_wandb(
      project = 'Architecture-2',
      name = f"Learning Rate:{training_params['learning_rate']} Batch Size:{training_params['batch_size']} Head: {num_head}",
      config = training_params
    )


    cnn_model = CNNSoundClassification(training_params['num_classes'])
    te_model = TransformerEncoder(training_params['model_dim'],
                                  training_params['num_layers'],
                                  training_params['ffn_hidden'],
                                  num_head,
                                  training_params['num_classes'],
                                  training_params['num_att_blocks']
                                  )


    cnnte_model = CNNTEModel(cnn_model, te_model, model_dim=training_params['model_dim'])

    # Initializing Model Trainer
    model_trainer = SoundClassifierTrainer(cnnte_model,
                                           esc_data.train_dataloader(),
                                           esc_data.val_dataloader(),
                                           esc_data.test_dataloader(),
                                           training_params['num_classes'],
                                           training_params['learning_rate'],
                                           using_wandb=True,
                                           early_stop=False,
                                           optimizer=training_params['optimizer'])

    print("\n\n")
    print(f"Results of model with Learning Rate:{training_params['learning_rate']} Batch Size:{training_params['batch_size']} Head: {num_head}\n")
    # Training
    model_trainer.train(training_params['epochs'])
    print("\n\n")
    wandb.finish()
    # Evaluation
    eval_outputs = model_trainer.evaluate()
    print("\n\n")
    print(f"F1-score: {eval_outputs['f1_score']:.2f}")
    print(f"Accuracy: {eval_outputs['accuracy']* 100:.2f}%")


    model_trainer.plot_confusion_matrix(eval_outputs['labels'], eval_outputs['predictions'], f"Confusion Matrix for Head:{num_head}")

    model_trainer.plot_auc_roc(eval_outputs['labels'], eval_outputs['probabilities'], training_params['num_classes'], f"AUC ROC for Head:{num_head}")

    model_trainer.plot_loss_and_accuracy_curves()

    model_trainer.report_parameters()

    print("\n\n")
