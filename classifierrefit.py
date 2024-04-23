import random
import numpy as np
import os
import math
from sklearn.metrics import classification_report
from sktime.base import load
from components.make_dataframes import make_dataframes
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.kernel_based import RocketClassifier
from pyinstrument import Profiler
from sklearn.metrics import ConfusionMatrixDisplay
from memory_profiler import profile
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

N_ESTIMATORS = 256
WINDOW_SIZE = 200
mem_bank: list = []


def memorable_bank(training_data, predicted_labels, training_labels, confidence_scores):
    confidence_threshold = 0.70  # Adjust this threshold based on your data
    memorable_bank: list = []
    for data, predicted_label, label, confidence in zip(
        training_data, predicted_labels, training_labels, confidence_scores
    ):
        # check if correctly classified and outlier based on confidence
        conf = np.max(confidence)
        nparr = np.array([data])
        if conf < (confidence_threshold) and predicted_label == label:
            memorable_bank.append((nparr, label))
    # add 6 random samples for each label in training data
    for i in range(len(set(training_labels))):
        for j in range(80):
            index = random.randint(0, len(training_data) - 1)
            if training_labels[index] == list(set(training_labels))[i]:
                nparr = np.array([training_data[index]])
                memorable_bank.append((nparr, training_labels[index]))
    return memorable_bank


@profile
def fitFirst(modelName: str, frames, labels, testFrames, testLabels, estimators=None):
    """
    Description of the function.

    Parameters:
    modelName (str): The name of the model to be saved.

    Returns:
    None
    """

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
╚═ N_ESTIMATORS: {estimators},  WINDOW_SIZE: {WINDOW_SIZE},  DATASIZE: {frames.shape[2]}, NUM_LABELS: {len(set(labels))}

"""
    )

    classifier = LSTMFCNClassifier(
        n_epochs=estimators,
        batch_size=50,
        dropout=0.55,
        verbose=0,
        filter_sizes=[64, 128, 256, 128, 64],
    )
    # classifier = CNNClassifier(
    #     n_epochs=estimators,
    #     batch_size=50,
    # )

    print("Fitting Classifier...")
    classifier.fit(frames, labels)
    global buffer

    y_pred = classifier.predict(testFrames)

    report = classification_report(testLabels, y_pred)
    fscoreSupportMicro = precision_recall_fscore_support(testLabels, y_pred, average="micro")
    ConfusionMatrixDisplay.from_predictions(testLabels, y_pred)
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

    print("\n\nF1 Score, Precision, Recall, Support (Micro):")
    print(fscoreSupportMicro)

    # Freeze the first 20 layers of the model (not sure if this works)
    # layers_to_freeze = 20
    # for layers in classifier.model_.layers[:layers_to_freeze]:
    #    layers.trainable = False
    global mem_bank
    mem_bank = memorable_bank(
        training_data=frames,
        predicted_labels=y_pred,
        training_labels=labels,
        confidence_scores=confidence_scores,
    )  # type: ignore
    plt.title(f"First fit of {str(classifier).split('(')[0]}")
    plt.savefig(f"./pictures/firstfit_{str(classifier).split('(')[0]}.svg", format="svg")
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


@profile
def learnNewLabel(
    modelName: str,
    folderName: str,
    frames,
    labels,
    testFrames,
    testLabels,
    testFramesIncludingBCM,
    testLabelsBCM,
    estimators=None,
):
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
╚═ N_ESTIMATORS: {N_ESTIMATORS},  WINDOW_SIZE: {WINDOW_SIZE},  DATASIZE: {frames.shape[2]}
"""
    )

    # Fit MiniRocket from SKTime using the frames and labels

    classifier = load(f"./models/{modelName}")

    labels_for_noisy_data = np.array([])
    noisey_data = []
    # if mem_bank:
    #     noise_factor = 0.1
    #     for data, label in mem_bank:
    #         noised_data = add_noise_and_padding(data, noise_factor, frames.shape[2])
    #         noisey_data.append(noised_data)
    #         labels_for_noisy_data = np.append(labels_for_noisy_data, label)

    # augmented_data = np.array(noisey_data)
    # # add augmented data to the training data ndarray, and the labels to the training labels ndarray
    # frames = np.concatenate((frames, augmented_data), axis=0)
    # labels = np.append(labels, labels_for_noisy_data)
    frames, testFramesIncludingBCM = minimize(frames, testFramesIncludingBCM)
    frames, labels = dataShuffler(frames, labels)
    print("Fitting Classifier...")
    classifier.fit(frames, labels)
    y_pred = classifier.predict(testFramesIncludingBCM)
    report = classification_report(testLabelsBCM, y_pred)
    fscoreSupportMicro = precision_recall_fscore_support(testLabelsBCM, y_pred, average="micro")
    ConfusionMatrixDisplay.from_predictions(testLabelsBCM, y_pred)

    # print("Predictions:")
    # print(y_pred)

    # print("\nActual: ")
    # print(testLabelsBCM)

    # print("\n\nProbabilities: ")
    # print(classifier.predict_proba(testFramesIncludingBCM))

    print("\n\nClassification Report:")
    print(report)

    plt.title(f"Refit of {str(classifier).split('(')[0]}, No Buffer")
    plt.savefig(f"./pictures/refit_{str(classifier).split('(')[0]}_nobuffer.svg", format="svg")
    print("\n\nF1 Score, Precision, Recall, Support (Micro):")
    print(fscoreSupportMicro)

    classifier.save(f"./models/{modelName}")


def dataShuffler(data, labels):
    combined = list(zip(data, labels))
    random.shuffle(combined)
    shuffledData, shuffledlabels = zip(*combined)
    shuffledData = np.array(shuffledData)
    shuffledlabels = np.array(shuffledlabels)
    return shuffledData, shuffledlabels


# frames, labels, testFrames, testLabels = make_dataframes("ProcessedData", WINDOW_SIZE)
profiler = Profiler()
profiler.start()
# fitFirst("lstmfcn", frames, labels, testFrames, testLabels, estimators=50)
# profiler.stop()
refitFrames, refitLabels, refitTestFrames, refitTestLabels = make_dataframes("NewData", WINDOW_SIZE)
stupidFrames, stupidLabels, stupidTestFrames, stupidTestLabels = make_dataframes("StupidTestData", WINDOW_SIZE)
learnNewLabel(
    "cnn",
    "NewData",
    frames=refitFrames,
    labels=refitLabels,
    testFramesIncludingBCM=stupidTestFrames,
    testLabelsBCM=stupidTestLabels,
    testFrames=None,
    testLabels=None,
    estimators=32,
)
profiler.stop()
profiler.print()
# prediction("CNN", "ProcessedData")
