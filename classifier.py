import os
import sys
import math
import numpy as np
import pandas as pd
from sktime.base import load
from sklearn.metrics import classification_report
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier


def make_dataframes(folder: str, isBuffer:bool = False):
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
    numberOfIterationsOverFilename = 0
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
            numberOfIterationsOverFilename = numberOfIterationsOverFilename + 1
            if(isBuffer and numberOfIterationsOverFilename == (math.ceil(0.1 * len(os.listdir(folder_path))))):
                break


    if (isBuffer is False):
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
                    time_series_data_Test.append(
                        [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                    )
                    activity_labels_Test.append(label)
    # make all sub-sub-arrays in the time_series_data array the same length as the smallest sub-sub-array
    min_length = min(
        len(subsubarray) for sublist in time_series_data_Training for subsubarray in sublist
    )
    min_length_test = min(len(subsubarray) for sublist in time_series_data_Test for subsubarray in sublist)
    min_length = min(min_length, min_length_test)
    time_series_data_Training = [
        [subsubarray[:min_length] for subsubarray in sublist]
        for sublist in time_series_data_Training
    ]
    time_series_data_Test = [
        [subsubarray[:min_length] for subsubarray in sublist]
        for sublist in time_series_data_Test
    ]
    activity_labels_Training = np.array(activity_labels_Training)
    time_series_data_Training = np.array(time_series_data_Training)
    activity_labels_Test = np.array(activity_labels_Test)
    time_series_data_Test = np.array(time_series_data_Test)
    return time_series_data_Training, activity_labels_Training, time_series_data_Test, activity_labels_Test

def fitFirst(modelName: str):
    print("Running firfirst -> Loading Data...")
    frames, labels, testFrames, testLabels = make_dataframes("ProcessedData")

    classifier = RocketClassifier()

    print("In progress: fitfirst -> Fitting Classifier...")
    classifier.fit(frames, labels)

    make_buffer_fitfirst(frames, labels)

    y_pred = classifier.predict(testFrames)

    report = classification_report(testLabels, y_pred)
    print("fitfirst -> Predictions:")
    print(y_pred)

    print("\nfitfirst -> Actual: ")
    print(testLabels)

    print("\n\nfitfirst -> Probabilities: ")
    print(classifier.predict_proba(testFrames))

    print("\n\nfitfirst -> Classification Report:")
    print(report)
    os.makedirs(f"./models", exist_ok=True)

    classifier.save(f"./models/{modelName}")


def refit(modelName: str, folderName: str):
    print("Running refit -> Loading Data...")
    frames, labels, testFrames, testLabels = make_dataframes(folderName)
    # Fit MiniRocket from SKTime using the frames and labels
    
    # Load existing buffer data
    buffer_filepath = "./buffer"
    buffer_frames = np.load(buffer_filepath + "/buffer_frames.npy")
    buffer_labels = np.load(buffer_filepath + "/buffer_labels.npy")

    # Append the buffer
    print("refit data shape init -> ", frames.shape, " - ", labels.shape)
    frames = minimize_and_concatenate(frames, buffer_frames)
    labels = np.append(labels, buffer_labels)
    print("refit data shape w/ buffer -> ", frames.shape, " - ", labels.shape)

    classifier = RocketClassifier.load_from_path(f"./models/{modelName}.zip")
    print("In progress: refit -> Fitting Classifier...")

    classifier.fit(frames, labels)

    print("refit -> Finished!")
    
    make_buffer_refit(frames, labels)

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

def make_buffer_fitfirst(_frames, _labels):
    #Percent in decimal
    buffer_percent = 0.25
    
    buffer_filepath = "./buffer"
    
    # Get unique labels and their counts
    unique_labels, label_counts = np.unique(_labels, return_counts=True)

    # Calculate the number of elements to select for each label
    ten_percent_counts = np.ceil(label_counts * buffer_percent).astype(int)

    # Create an array to store the selected labels and frames
    selected_frames = []
    selected_labels = []

    # Iterate over unique labels and select 10% of each
    for label, count in zip(unique_labels, ten_percent_counts):
        indices = np.where(_labels == label)[0]  # Get indices of current label
        selected_indices = indices[:count]  # Select 10% of indices
        selected_frames.extend(_frames[selected_indices])  # Add selected labels to the result
        selected_labels.extend(_labels[selected_indices])  # Add selected labels to the result

    np.save(buffer_filepath + "/buffer_frames", selected_frames)
    np.save(buffer_filepath + "/buffer_labels", selected_labels)

    #print("10 percent of labels from first fit: ", selected_labels)

def make_buffer_refit(_frames, _labels):
    #Percent in decimal
    buffer_percent = 0.25
    
    buffer_filepath = "./buffer"
    
    # Load existing buffer data
    buffer_frames_old = np.load(buffer_filepath + "/buffer_frames.npy")
    buffer_labels_old = np.load(buffer_filepath + "/buffer_labels.npy")

    print("Pre refit buffer labels: ", buffer_labels_old)

    # Slice the original array to get the first 10% of elements
    percent_length = math.ceil(int(len(_frames) * buffer_percent))
    percent_length = math.ceil(int(len(_labels) * buffer_percent))

    buffer_frames = _frames[:percent_length]
    buffer_labels = _labels[:percent_length]

    # Check if the current label is already in the buffer files
    if buffer_labels[0] in buffer_labels_old:
        # buffer_buffer_frames = []
        # buffer_buffer_labels = []
        # i = 0

        # for label in buffer_labels_old:
        #     if label != buffer_labels[0]:
        #         buffer_buffer_frames = minimize_and_concatenate(buffer_buffer_frames, buffer_frames_old[i])
        #         buffer_buffer_labels.append(label)

        #     i += 1

        # buffer_buffer_labels = np.append(buffer_buffer_labels, buffer_labels)
        # buffer_buffer_frames = minimize_and_concatenate(buffer_buffer_frames, buffer_frames)
        
        # buffer_labels_old = buffer_buffer_labels
        # buffer_frames_old = buffer_buffer_frames

        # print("Updated existing entries in buffer: ", buffer_buffer_labels)

        pass
    else:
        buffer_labels_old = np.append(buffer_labels_old, buffer_labels)
        buffer_frames_old = minimize_and_concatenate(buffer_frames_old, buffer_frames)

    print("Saved new buffer labels: ", buffer_labels_old)

    if not os.path.exists(buffer_filepath):
        print("Run fitfirst() first to create buffer files!")

    np.save(buffer_filepath + "/buffer_frames", buffer_frames_old)
    np.save(buffer_filepath + "/buffer_labels", buffer_labels_old)

def minimize_and_concatenate(arr1, arr2):
    """
    Trim both arrays along the third dimension to the minimum length,
    and concatenate them along the first dimension.
    
    Parameters:
        arr1 (numpy.ndarray): First array.
        arr2 (numpy.ndarray): Second array.
    
    Returns:
        numpy.ndarray: Concatenated array with the third dimension trimmed.
    """
    # Get the minimum length among the two arrays along the third dimension
    min_length = min(arr1.shape[2], arr2.shape[2])
    
    # Trim both arrays along the third dimension to the minimum length
    trimmed_arr1 = arr1[:, :, :min_length]
    trimmed_arr2 = arr2[:, :, :min_length]
    
    # Concatenate the trimmed arrays along the first dimension
    concatenated_array = np.concatenate((trimmed_arr1, trimmed_arr2), axis=0)

    print("Frame Minimization and Concatenation Finished!")

    return concatenated_array

# make_dataframes("ProcessedData", True)
# fitFirst("Rocket")
refit("Rocket", "NewData")
# prediction("Rocket", "ProcessedData")