import os
import pandas as pd
import numpy as np
import random


def make_dataframes(folder: str, window_size: int = 100):
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
                time_series = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]

                # Window the time series
                for i in range(0, len(accel_x) - window_size + 1, window_size):
                    windowed_time_series = [subseries[i : i + window_size] for subseries in time_series]
                    time_series_data_Training.append(windowed_time_series)
                    activity_labels_Training.append(label)

    for filename in sorted(os.listdir(folder_path_test)):
        if filename.endswith(".csv") and not filename.startswith(".~"):
            file_path = os.path.join(folder_path_test, filename)

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
                time_series = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]

                # Window the time series
                for i in range(0, len(accel_x) - window_size + 1, window_size):
                    windowed_time_series = [subseries[i : i + window_size] for subseries in time_series]
                    time_series_data_Test.append(windowed_time_series)
                    activity_labels_Test.append(label)
    # make all sub-sub-arrays in the time_series_data array the same length as the smallest sub-sub-array
    min_length = min(len(subsubarray) for sublist in time_series_data_Training for subsubarray in sublist)
    min_length_test = min(len(subsubarray) for sublist in time_series_data_Test for subsubarray in sublist)
    min_length = min(min_length, min_length_test)
    time_series_data_Training = [
        [subsubarray[:min_length] for subsubarray in sublist] for sublist in time_series_data_Training
    ]
    combined = list(zip(time_series_data_Training, activity_labels_Training))
    random.shuffle(combined)
    time_series_data_Training, activity_labels_Training = zip(*combined)
    time_series_data_Test = [[subsubarray[:min_length] for subsubarray in sublist] for sublist in time_series_data_Test]
    activity_labels_Training = np.array(activity_labels_Training)
    time_series_data_Training = np.array(time_series_data_Training)
    activity_labels_Test = np.array(activity_labels_Test)
    time_series_data_Test = np.array(time_series_data_Test)
    return time_series_data_Training, activity_labels_Training, time_series_data_Test, activity_labels_Test