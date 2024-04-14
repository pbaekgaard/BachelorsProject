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


def make_classifier(
    time_series_data_train,
    activity_labels_train,
    time_series_data_test,
    activity_labels_test,
    learning_rate=0.01,
    epochs=N_EPOCHS,
):
    """
    Builds, trains, and evaluates an LSTM classifier for activity recognition.

    Args:
        time_series_data_train (np.ndarray): Training data features.
        activity_labels_train (np.ndarray): Training data labels.
        time_series_data_test (np.ndarray): Testing data features.
        activity_labels_test (np.ndarray): Testing data labels.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        epochs (int, optional): Number of training epochs. Defaults to 10.

    Returns:
        None
    """
    # Convert character labels to numerical labels (assuming unique characters)
    label_map_index = {i: label for i, label in enumerate(set(activity_labels_train))}
    label_map = {label: i for i, label in enumerate(set(activity_labels_train))}
    activity_labels_train_num = np.array(
        [label_map[label] for label in activity_labels_train]
    )
    activity_labels_test_num = np.array(
        [label_map[label] for label in activity_labels_test]
    )

    # Convert data to Tensors
    train_data = TensorDataset(
        torch.from_numpy(time_series_data_train).float(),
        torch.from_numpy(activity_labels_train_num),
    )
    test_data = TensorDataset(
        torch.from_numpy(time_series_data_test).float(),
        torch.from_numpy(activity_labels_test_num),
    )

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Define the LSTM model
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, batch_first=True, dropout=0.75, num_layers=6
            )
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            self.lstm.flatten_parameters()
            _, (hidden, _) = self.lstm(x)
            out = hidden[-1]
            return self.classifier(out)

    # Define model parameters
    input_size = len(
        time_series_data_train[0][0]
    )  # Number of features in each sequence
    hidden_size = 256  # Number of hidden units in LSTM
    num_classes = len(set(activity_labels_train))  # Number of unique activity labels

    # Create the model instance
    model = LSTMClassifier(input_size, hidden_size, num_classes)

    # Use cross-entropy loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        for i, (data, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

    # Evaluate the model on test data
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            outputs = model(data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if len(predicted) > 5:
                for i in range(len(predicted)):
                    prob_list = probabilities[i].tolist()
                    print(
                        f"Sample {i+1} - Predicted: {label_map_index[predicted[i].item()]} - Probability: {prob_list}"
                    )
        predicted_labels = predicted.numpy()
        predicted_labels_char = [label_map_index[label] for label in predicted_labels]
        print(f"Predicted labels: {predicted_labels_char}")
        print(f"Actual labels: {activity_labels_test}")
        print(f"Accuracy of the model on the test data: {100 * correct / total:.3f}%")


def pytorchTest():
    (
        time_series_data_train,
        activity_labels_train,
        time_series_data_test,
        activity_labels_test,
    ) = make_dataframes("ProcessedData")
    # Train and evaluate the classifier
    make_classifier(
        time_series_data_train,
        activity_labels_train,
        time_series_data_test,
        activity_labels_test,
    )


pytorchTest()


def fitFirst(modelName: str):
    """
    Description of the function.

    Parameters:
    modelName (str): The name of the model to be saved.

    Returns:
    None
    """

    print("Loading Data...")
    frames, labels, testFrames, testLabels = make_dataframes("ProcessedData")

    classifier = CNNClassifier(n_epochs=50, verbose=True)
    if testFrames == frames:
        print("oh no")
    print("Fitting Classifier...")
    classifier.fit(frames, labels)

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


# fitFirst("CNN")
# refit("CNN", "newData")
# prediction("CNN", "ProcessedData")
