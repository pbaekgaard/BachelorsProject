import random
import numpy as np
import os
import math
from sklearn.metrics import classification_report
from sktime.base import load
from components.make_dataframes import make_dataframes
from sktime.classification.interval_based import DrCIF
import pandas as pd

N_ESTIMATORS = 256
WINDOW_SIZE = 300
mem_bank: list = []


def memorable_bank(training_data, predicted_labels, training_labels, confidence_scores):
    confidence_threshold = 0.95  # Adjust this threshold based on your data
    memorable_bank: list = []
    for data, predicted_label, label, confidence in zip(
        training_data, predicted_labels, training_labels, confidence_scores
    ):
        # check if correctly classified and outlier based on confidence
        conf = np.max(confidence)
        nparr = np.array([data])
        if conf < (confidence_threshold) and predicted_label == label:
            memorable_bank.append((nparr, label))
    return memorable_bank


def fitFirst(modelName: str, estimators: int = None):
    """
    Description of the function.

    Parameters:
    modelName (str): The name of the model to be saved.

    Returns:
    None
    """

    frames, labels, testFrames, testLabels = make_dataframes("ProcessedData", WINDOW_SIZE)
    if estimators is None or estimators == 0:
        estimators = int((WINDOW_SIZE * (2 / (math.pi * 1.5))))
        if estimators > 256:
            estimators = 256
    # print fitting first moddel in ascii art
    print(
        f"""
███████╗██╗████████╗████████╗██╗███╗   ██╗ ██████╗     ███████╗██╗██████╗ ███████╗████████╗
██╔════╝██║╚══██╔══╝╚══██╔══╝██║████╗  ██║██╔════╝     ██╔════╝██║██╔══██╗██╔════╝╚══██╔══╝
█████╗  ██║   ██║      ██║   ██║██╔██╗ ██║██║  ███╗    █████╗  ██║██████╔╝███████╗   ██║
██╔══╝  ██║   ██║      ██║   ██║██║╚██╗██║██║   ██║    ██╔══╝  ██║██╔══██╗╚════██║   ██║
██║     ██║   ██║      ██║   ██║██║ ╚████║╚██████╔╝    ██║     ██║██║  ██║███████║   ██║
╚═╝     ╚═╝   ╚═╝      ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝     ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝    
╚══════ N_ESTIMATORS: {estimators},  WINDOW_SIZE: {WINDOW_SIZE},  DATASIZE: {frames.shape[2]}, NUM_CLASSES = 6

"""
    )
    classifier = DrCIF(
        n_estimators=estimators,
        att_subsample_size=6,
        n_jobs=-1,
        random_state=44,
    )
    print("Fitting Classifier...")
    classifier.fit(frames, labels)
    global buffer

    y_pred = classifier.predict(testFrames)

    report = classification_report(testLabels, y_pred)
    # print("Predictions:")
    # print(y_pred)

    # print("\nActual: ")
    # print(testLabels)

    # print("\n\nProbabilities: ")
    confidence_scores = classifier.predict_proba(testFrames)
    # print(confidence_scores)

    print("\n\nClassification Report:")
    print(report)
    os.makedirs(f"./models", exist_ok=True)

    # Freeze the first 20 layers of the model (not sure if this works)
    # layers_to_freeze = 20
    # for layers in classifier.model_.layers[:layers_to_freeze]:
    #    layers.trainable = False
    global mem_bank
    mem_bank = memorable_bank(training_data=frames, predicted_labels=y_pred, training_labels=labels, confidence_scores=confidence_scores)  # type: ignore
    classifier.save(f"./models/{modelName}")


def add_noise_and_padding(data, noise_level, currentFrameLength):
    """
    Adds Gaussian noise to a time series data point.

    Args:
        data: A numpy array representing the time series data.
        noise_level: A float representing the standard deviation of the noise.

    Returns:
        A numpy array with noise added to the original data.
    """

    # Generate Gaussian noise with the same shape as the data
    noise = np.random.normal(scale=noise_level, size=data.shape)
    # Add noise to the data
    noisy_data = data + noise

    # pad the data to the same length as the current frame length
    if noisy_data.shape[2] < currentFrameLength:
        padding = np.zeros((noisy_data.shape[0], noisy_data.shape[1], currentFrameLength - noisy_data.shape[2]))
        noisy_data = np.concatenate((noisy_data, padding), axis=2)
    # Reshape back to the original shape if needed (assuming single time series)
    return noisy_data.squeeze()


def minimize(frames, tests):
    min_length = min(frames.shape[2], tests.shape[2])
    frames = frames[:, :, :min_length]
    tests = tests[:, :, :min_length]
    return frames, tests


def learnNewLabel(modelName: str, folderName: str, estimators: int = None):
    frames, labels, testFrames, testLabels = make_dataframes(folderName, WINDOW_SIZE)
    framesIncludingBCM, labelsBCM, testFramesIncludingBCM, testLabelsBCM = make_dataframes(
        "StupidTestData", WINDOW_SIZE
    )
    if estimators is None or estimators == 0:
        estimators = int((WINDOW_SIZE * (2 / (math.pi * 1.5))))
        if estimators > 256:
            estimators = 256
    print(
        f"""
██████╗ ███████╗███████╗██╗████████╗████████╗██╗███╗   ██╗ ██████╗
██╔══██╗██╔════╝██╔════╝██║╚══██╔══╝╚══██╔══╝██║████╗  ██║██╔════╝
██████╔╝█████╗  █████╗  ██║   ██║      ██║   ██║██╔██╗ ██║██║  ███╗
██╔══██╗██╔══╝  ██╔══╝  ██║   ██║      ██║   ██║██║╚██╗██║██║   ██║
██║  ██║███████╗██║     ██║   ██║      ██║   ██║██║ ╚████║╚██████╔╝
╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝   ╚═╝      ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝
╚══════ N_ESTIMATORS: {N_ESTIMATORS},  WINDOW_SIZE: {WINDOW_SIZE},  DATASIZE: {frames.shape[2]}
"""
    )

    # Fit MiniRocket from SKTime using the frames and labels

    classifier = load(f"./models/{modelName}")

    labels_for_noisy_data = np.array([])
    noisey_data = []
    if mem_bank:
        noise_factor = 0.1
        for data, label in mem_bank:
            noised_data = add_noise_and_padding(data, noise_factor, frames.shape[2])
            noisey_data.append(noised_data)
            labels_for_noisy_data = np.append(labels_for_noisy_data, label)

    augmented_data = np.array(noisey_data)
    # add augmented data to the training data ndarray, and the labels to the training labels ndarray
    frames = np.concatenate((frames, augmented_data), axis=0)
    labels = np.append(labels, labels_for_noisy_data)
    frames, testFramesIncludingBCM = minimize(frames, testFramesIncludingBCM)
    print("Fitting Classifier...")
    classifier.fit(frames, labels)
    y_pred = classifier.predict(testFramesIncludingBCM)
    report = classification_report(testLabelsBCM, y_pred)
    # print("Predictions:")
    # print(y_pred)

    # print("\nActual: ")
    # print(testLabelsBCM)

    # print("\n\nProbabilities: ")
    # print(classifier.predict_proba(testFramesIncludingBCM))

    print("\n\nClassification Report:")
    print(report)

    classifier.save(f"./models/{modelName}")


fitFirst("minirocket")
learnNewLabel("minirocket", "NewData")
# prediction("CNN", "ProcessedData")
