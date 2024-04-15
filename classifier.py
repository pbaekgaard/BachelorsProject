import pandas as pd
import numpy as np
import os
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier

from sklearn.metrics import classification_report


def make_dataframes(folder: str):
    # Path to the ProcessedData folder
    base_path = folder
    folder_path = os.path.join(base_path, "Training")
    folder_path_test = os.path.join(base_path, "Test")

    # Initialize empty lists to store time series data and labels
    time_series_data_Training = []
    activity_labels_Training = []
    time_series_data_Test = []
    activity_labels_Test = []


    # Iterate over each CSV file in the specified folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".csv") and not filename.startswith(".~"):
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

            # Group the DataFrame by 'Activity Label'
            grouped = df.groupby("Activity Label")

            # Iterate over each group (activity label)
            for label, group in grouped:
                # Extract relevant columns (excluding 'Subject-Id' and 'Time stamp')
                accel_x = group["Accel_x"].values.tolist()
                accel_y = group["Accel_y"].values.tolist()
                accel_z = group["Accel_z"].values.tolist()
                gyro_x = group["Gyro_x"].values.tolist()
                gyro_y = group["Gyro_y"].values.tolist()
                gyro_z = group["Gyro_z"].values.tolist()
                time_series_data_Training.append(
                    [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                )
                activity_labels_Training.append(label)

    for filename in sorted(os.listdir(folder_path_test)):
        if filename.endswith(".csv") and not filename.startswith(".~"):
            file_path = os.path.join(folder_path_test, filename)

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

            # Group the DataFrame by 'Activity Label'
            grouped = df.groupby("Activity Label")

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
    classifier = RocketClassifier.load_from_path(f"./models/{modelName}.zip")
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


fitFirst("CNN")
# refit("Rocket", "newData")
# prediction("Rocket", "ProcessedData")
