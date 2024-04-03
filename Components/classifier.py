import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from sktime.classification.kernel_based import RocketClassifier
from sklearn.model_selection import train_test_split
import sktime.datatypes as skdtypes
import numpy as np
from sktime.classification.hybrid import HIVECOTEV2
from sklearn.metrics import classification_report

SAMPLELEN = 600


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


def split_file(file_path, file, typeOfData):
    dataPaths = get_data_paths(file_path)
    dataFromFile = load_data_from_file(
        os.path.abspath(os.path.join(dataPaths["Accel"][typeOfData], file))
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
    XTrain_Combined = np.empty((0, 3, SAMPLELEN))
    YTrain_Combined = []
    for filename in dataFileArrayAccel:
        if filename.startswith(".~") == False:
            XTrain, YTrain = split_file(file_path, filename, typeOfData)
            XTrain_Combined = np.concatenate((XTrain_Combined, XTrain), axis=0)
            YTrain_Combined.extend(YTrain)
    XTrain_Combined = np.array(XTrain_Combined)
    YTrain_Combined = np.array(YTrain_Combined)
    return XTrain_Combined, YTrain_Combined


def load_data_test(file_path, typeOfData):
    dataPaths = get_data_paths(file_path)
    dataFileArrayAccel = os.listdir(dataPaths["Accel"][typeOfData])
    XTrain_Combined = np.empty((0, 3, SAMPLELEN))
    YTrain_Combined = []
    for filename in dataFileArrayAccel:
        if filename.startswith(".~") == False:
            print(f"getting testdata from file: {filename}")
            XTrain, YTrain = split_file(file_path, filename, typeOfData)
            XTrain_Combined = np.concatenate((XTrain_Combined, XTrain), axis=0)
            YTrain_Combined.extend(YTrain)

    XTrain_Combined = np.array(XTrain_Combined)
    YTrain_Combined = np.array(YTrain_Combined)
    return XTrain_Combined, YTrain_Combined


classifier = HIVECOTEV2(
    drcif_params={"n_estimators": 200},  # Set number of estimators for DrCIF
    time_limit_in_minutes=4,  # Set a time limit of 2 minutes
)
XTrain, YTrain = load_data("ProcessedData", "Training")
XTest, YTest = load_data_test("ProcessedData", "Test")

classifier.fit(XTrain, YTrain)
print("running classifier.fit")
y_pred = classifier.predict(XTest)
y_predproba = classifier.predict_proba(XTest)
print(y_pred)
print(y_predproba)

report = classification_report(YTest, y_pred)
print("Classification Report:\n", report)
