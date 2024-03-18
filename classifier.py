import pandas as pd
import os
from sktime.classification.kernel_based import RocketClassifier
from sklearn.model_selection import train_test_split
import sktime.datatypes as skdtypes

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

def load_data(file_path, typeOfData):
    # prepath = os.path.abspath(file_path)
    # dataFileArray = [entry.path for entry in os.scandir(prepath) if entry.is_file()]
    dataPaths = get_data_paths(file_path)
    dataFileArrayAccel = os.listdir(dataPaths["Accel"][typeOfData])
    dataFileArrayGyro = os.listdir(dataPaths["Gyro"][typeOfData])
    TrainingAccel = []
    TrainingGyro = []
    for filename in dataFileArrayAccel:
        dataFromFile = load_data_from_file(os.path.abspath(os.path.join(dataPaths["Accel"][typeOfData],filename)))
        features = dataFromFile[["Timestamp","x", "y", "z"]]
        targetVariable = dataFromFile["Activity Label"]
        TrainingAccel.append((features, targetVariable))

    for filename in dataFileArrayGyro:
        dataFromFile = load_data_from_file(os.path.abspath(os.path.join(dataPaths["Gyro"][typeOfData],filename)))
        features = dataFromFile[["x", "y", "z"]]
        targetVariable = dataFromFile["Activity Label"]
        TrainingGyro.append((features, targetVariable))

    trainingData = {"Accel": TrainingAccel,"Gyro": TrainingGyro}
    return trainingData
    

trainingData = load_data("ProcessedData", "Training")
# print(trainingData["Accel"][0][0])
classifier = RocketClassifier(use_multivariate="yes")
testData = load_data("ProcessedData", "Test")
for entry in trainingData["Accel"]:
    features, targetVariable = entry
    X_train = features
    print(X_train)
    y_train = targetVariable
    print(y_train)

    # Split X_train DataFrame into a list of pandas DataFrames
    X_train_list = [X_train.iloc[i:i+1] for i in range(len(X_train))]

    # Run sktime's check_raise function to diagnose the input format issue
    skdtypes.check_raise(X_train_list, "df-list")

    # Fit the classifier with the training data
    classifier.fit(X_train_list, y_train)

for entry in testData["Accel"]:
   features, targetVariable = entry
   X_test = features

   X_test_list = [X_test.iloc[i:i+1] for i in range(len(X_test))]
   y_pred = classifier.predict(X_test_list)
   print(y_pred)

print("Classifier fit test: ", classifier.is_fitted)
#y_pred = classifier.predict(testData["Accel"][0])

# X_train = trainingData["Accel"][0]
# y_train = trainingData["Accel"][1]

# classifier = RocketClassifier(use_multivariate="yes")
# classifier.fit(X_train, y_train)