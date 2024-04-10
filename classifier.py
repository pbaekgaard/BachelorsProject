import pandas as pd
import numpy as np
import os
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
from sklearn.metrics import classification_report


def make_dataframes(folder: str, type: str):
    # Path to the ProcessedData folder
    base_path = folder
    folder_path = os.path.join(base_path, type)

    # Initialize empty lists to store time series data and labels
    time_series_data = []
    activity_labels = []

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
                time_series_data.append(
                    [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                )
                activity_labels.append(label)
    # make all sub-sub-arrays in the time_series_data array the same length as the smallest sub-sub-array
    min_length = min(
        len(subsubarray) for sublist in time_series_data for subsubarray in sublist
    )
    time_series_data = [
        [subsubarray[:min_length] for subsubarray in sublist]
        for sublist in time_series_data
    ]
    activity_labels = np.array(activity_labels)
    time_series_data = np.array(time_series_data)
    return time_series_data, activity_labels


def fitFirst(modelName: str):
    print("Loading Data...")
    frames, labels = make_dataframes("ProcessedData", "Training")

    # Fit MiniRocket from SKTime using the frames and labels

    testFrames, testLabels = make_dataframes("ProcessedData", "Test")

    classifier = CNNClassifier()
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

    classifier.save(f"./models/{modelName}")


def refit(modelName: str, folderName: str):
    print("Loading Data...")
    frames, labels = make_dataframes(folderName, "Training")
    # Fit MiniRocket from SKTime using the frames and labels

    classifier = RocketClassifier.load_from_path(f"./models/{modelName}.zip")
    print("Fitting Classifier...")
    classifier.fit(frames, labels)
    classifier.save(f"./models/{modelName}")

def prediction(modelName: str, folderName: str):
    testFrames, testLabels = make_dataframes(folderName, "Test")
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


