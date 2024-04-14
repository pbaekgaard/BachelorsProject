import random
import pandas as pd
import numpy as np
import os
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
from sklearn.metrics import classification_report
from sktime.base import load
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from components.make_dataframes import make_dataframes

N_EPOCHS = 250
BATCH_SIZE = 32


def fitFirst(modelName: str):
    """
    Description of the function.

    Parameters:
    modelName (str): The name of the model to be saved.

    Returns:
    None
    """

    class ReplayBuffer:
        def __init__(self, max_size):
            self.max_size = max_size
            self.buffer = []

        def add(self, data):
            if len(self.buffer) >= self.max_size:
                self.buffer.pop(0)  # Remove oldest entry if full
            self.buffer.append(data)

        def sample(self, batch_size):
            return random.sample(self.buffer, batch_size)

    print("Loading Data...")
    frames, labels, testFrames, testLabels = make_dataframes("ProcessedData")

    classifier = RocketClassifier(
        rocket_transform="minirocket",
        n_features_per_kernel=6,
        use_multivariate="yes",
        random_state=42,
    )

    print("Fitting Classifier...")
    classifier.fit(frames, labels)
    global buffer
    buffer = ReplayBuffer(max_size=1000)

    for epoch in range(N_EPOCHS):
        buffer.add((frames[epoch], labels[epoch]))

    y_pred = classifier.predict(testFrames)

    report = classification_report(testLabels, y_pred)
    print("Predictions:")
    print(y_pred)

    print("\nActual: ")
    print(testLabels)

    print("\n\nProbabilities: ")
    print(classifier.predict_proba(testFrames))

    print("\n\nClassification Report:")
    print(report)
    os.makedirs(f"./models", exist_ok=True)

    # Freeze the first 20 layers of the model (not sure if this works)
    # layers_to_freeze = 20
    # for layers in classifier.model_.layers[:layers_to_freeze]:
    #    layers.trainable = False
    classifier.save(f"./models/{modelName}")


def refit(modelName: str, folderName: str):
    print("Loading Data...")
    frames, labels, testFrames, testLabels = make_dataframes(folderName)
    # Fit MiniRocket from SKTime using the frames and labels
    classifier = RocketClassifier.load_from_path(f"./models/{modelName}.zip")
    print("Fitting Classifier...")

    buffer_data = buffer.sample(BATCH_SIZE)
    frames = np.concatenate((frames, buffer_data[0]))
    labels = np.concatenate((labels, buffer_data[1]))
    classifier.fit(frames, labels)

    classifier.save(f"./models/{modelName}")


def prediction(modelName: str, folderName: str):
    frames, labels, testFrames, testLabels = make_dataframes(folderName)
    classifier = load(f"./models/{modelName}")
    y_pred = classifier.predict(testFrames)
    report = classification_report(testLabels, y_pred)
    print("Predictions:")
    print(y_pred)

    print("\nActual: ")
    print(testLabels)

    print("\n\nProbabilities: ")
    print(classifier.predict_proba(testFrames))

    print("\n\nClassification Report:")
    print(report)


fitFirst("minirocket")
refit("minirocket", "newData")
# prediction("CNN", "ProcessedData")
