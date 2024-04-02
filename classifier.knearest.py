import pandas as pd
import os
from sktime.classification.kernel_based import RocketClassifier
import numpy as np
from sklearn.metrics import classification_report
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
SAMPLELEN = 900


def get_data_paths(folder_path):
    data_paths = {}
    for sensor in ["Accel", "Gyro"]:
        sensor_path = f"{folder_path}/{sensor}"
        data_paths[sensor] = {
            "Training": f"{sensor_path}/Training/",
            "Validation": f"{sensor_path}/Validation/",
            "Test": f"{sensor_path}/Test/",
        }
    return data_paths


def load_data_from_file(file_path):
    data = pd.read_csv(file_path)
    return data


def split_file(file_path, file, typeOfData, device):
    dataPaths = get_data_paths(file_path)
    dataFromFile = load_data_from_file(
        os.path.abspath(os.path.join(dataPaths[device][typeOfData], file))
    )

    features = dataFromFile[["Activity Label", "Timestamp", "x", "y", "z"]]
    Y_train = []
    X_train = []
    for i in range(0, len(features.index) // 100 * 100, SAMPLELEN):
        previousLabel = ""
        x = []
        y = []
        z = []
        for k in range(0, SAMPLELEN):
            if (i + k) < len(features.index):
                if previousLabel == "" or features.iloc[i + k][0] == previousLabel:
                    x.append(features.iloc[i + k][2])
                    y.append(features.iloc[i + k][3])
                    z.append(features.iloc[i + k][4])
                    previousLabel = features.iloc[i + k][0]
                else:
                    break
        if len(x) == len(y) == len(z) == SAMPLELEN:
            X_train.append([x, y, z])
            Y_train.append(previousLabel)
    Y_train = np.array(Y_train)
    X_train = np.array(X_train)
    # Reshape XTrain to be in numpy3D format
    X_train = X_train.reshape(-1, 3, SAMPLELEN)

    return X_train, Y_train


def load_data(file_path, typeOfData):
    dataPaths = get_data_paths(file_path)
    dataFileArrayAccel = os.listdir(dataPaths["Accel"][typeOfData])
    dataFileArrayGyro = os.listdir(dataPaths["Gyro"][typeOfData])
    XTrain_Combined_Accel = np.empty((0, 3, SAMPLELEN))
    XTrain_Combined_Gyro = np.empty((0, 3, SAMPLELEN))
    YTrain_Combined = []
    for filename in dataFileArrayAccel:
        if filename.startswith(".~") == False:
            XTrain, YTrain = split_file(file_path, filename, typeOfData, "Accel")
            XTrain_Combined_Accel = np.concatenate(
                (XTrain_Combined_Accel, XTrain), axis=0
            )
            YTrain_Combined.extend(YTrain)

    for filename in dataFileArrayGyro:
        if filename.startswith(".~") == False:
            XTrainGyro, YTrainGyro = split_file(file_path, filename, typeOfData, "Gyro")
            XTrain_Combined_Gyro = np.concatenate((XTrain_Combined_Gyro, XTrainGyro))

    XTrain_Combined_Accel = np.array(XTrain_Combined_Accel)
    XTrain_Combined_Gyro = np.array(XTrain_Combined_Gyro)
    min_length = min(XTrain_Combined_Accel.shape[0], XTrain_Combined_Gyro.shape[0])
    XTrain_Combined = np.concatenate(
        (XTrain_Combined_Accel[:min_length], XTrain_Combined_Gyro[:min_length]), axis=1
    )
    YTrain_Combined = np.array(YTrain_Combined[:min_length])
    return XTrain_Combined, YTrain_Combined


print("Loading data..\n")
XTest, YTest = load_data("ProcessedData", "Test")

XTrain, YTrain = load_data("ProcessedData", "Training")

classifier = KNeighborsTimeSeriesClassifier(n_neighbors=3)

print("Fitting Classifier..\n")
classifier.fit(XTrain, YTrain)

print("Running Prediction..\n")
y_pred = classifier.predict(XTest)
y_predproba = classifier.predict_proba(XTest)
print(f"guesses: \n {y_pred}")
print(f"Probabilities from guess: \n {y_predproba}")
print(f"Actual: \n {YTest}")

report = classification_report(YTest, y_pred)
print("Classification Report:\n", report)

classifier.save("./models/CNN")
