import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Function to load the time series data and activity labels from the specified folder
def make_sequences(folder: str, label_encoder=None) -> tuple[tuple, tuple, LabelEncoder]:
    """
    Function loads the time series data and activity labels from the specified folder.

    Parameters:
    folder (str): The path to the folder containing the CSV files.

    Returns:
    time_series_data_Training (np.ndarray): An array containing the time series data for training.

    activity_labels_Training (np.ndarray): An array containing the activity labels for training.

    time_series_data_Test (np.ndarray): An array containing the time series data for testing.

    activity_labels_Test (np.ndarray): An array containing the activity labels for testing.
    """
    # Path to the ProcessedData folder
    base_path = folder
    folder_path = os.path.join(base_path, "Training")
    folder_path_test = os.path.join(base_path, "Test")

    # Initialize empty lists to store time series data and labels

    if label_encoder is None:
        label_encoder = LabelEncoder()

    sequences = []
    sequences_test = []
    FEATURE_COLUMNS = None
    # Iterate over each CSV file in the specified folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".csv") and not filename.startswith(".~"):
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)
            FEATURE_COLUMNS = df.columns.tolist()[3:]
            # Group the DataFrame by 'Activity Label'
            grouped = df.groupby("Activity Label")

            # Iterate over each group (activity label)
            for label, group in grouped:
                # Extract relevant columns (excluding 'Subject-Id' and 'Time stamp')
                sequence_features = group[FEATURE_COLUMNS]
                sequences.append((sequence_features, label))
    for filename in sorted(os.listdir(folder_path_test)):
        if filename.endswith(".csv") and not filename.startswith(".~"):
            file_path = os.path.join(folder_path_test, filename)

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)
            FEATURE_COLUMNS = df.columns.tolist()[3:]
            # Group the DataFrame by 'Activity Label'
            grouped = df.groupby("Activity Label")
            # Iterate over each group (activity label)
            for label, group in grouped:
                # Extract relevant columns (excluding 'Subject-Id' and 'Time stamp')
                sequence_features_test = group[FEATURE_COLUMNS]
                sequences_test.append((sequence_features_test, label))
    # make all sub-sub-arrays in the time_series_data array the same length as the smallest sub-sub-array
    sequences = minimize(sequences)
    sequences_test = minimize(sequences_test)
    sequences, label_encoder = encode_labels(sequences, label_encoder)
    sequences_test, label_encoder = encode_labels(sequences_test, label_encoder)
    return (sequences, sequences_test, label_encoder)


def encode_labels(seq, label_encoder):

    labels = [label for _, label in seq]
    encoded_labels = []
    if hasattr(label_encoder, "classes_") is False:
        encoded_labels = label_encoder.fit_transform(labels)
    else:
        for label in labels:
            if label in label_encoder.classes_:
                encoded_labels.append(label_encoder.transform([label])[0])
            else:
                encoded_labels.append(len(label_encoder.classes_))  # Assign a new integer label
                label_encoder.classes_ = np.append(label_encoder.classes_, label)
    encoded_sequence_array = [(arr[0], encoded_label) for arr, encoded_label in zip(seq, encoded_labels)]
    return encoded_sequence_array, label_encoder


def minimize(seq):
    min_length = min(len(arr) for arr, _ in seq)
    result_array = []
    for arr, value in seq:
        result_array.append((arr[:min_length], value))
    return result_array
