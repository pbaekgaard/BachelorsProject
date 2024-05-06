import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from random import uniform
import random



import seaborn as sns
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

SAMPLELEN = 10


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
#XTest, YTest = load_data("ProcessedData", "Test")

#print(XTest)




def euclidean_matrix(data, centroids):
    """
    Compute the Euclidean distance between each data point and each centroid.
    Data should be an (n, m) matrix, centroids an (k, m) matrix.
    Returns an (n, k) matrix of distances.
    """

    # Broadcasting and vectorized subtraction and squaring to compute distances
    return np.sqrt(np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2))

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X_train):
        # Initialize centroids using k-means++ method
        initial_idx = random.randint(0, X_train.shape[0] - 1)
        self.centroids = X_train[initial_idx:initial_idx+1, :]
        for _ in range(1, self.n_clusters):
            dists = euclidean_matrix(X_train, self.centroids)
            closest_dist_sq = np.min(dists, axis=1)
            probabilities = closest_dist_sq / closest_dist_sq.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            r = random.random()
            new_centroid_idx = np.searchsorted(cumulative_probabilities, r)
            self.centroids = np.vstack([self.centroids, X_train[new_centroid_idx, :]])

        # Iteratively update centroids
        for _ in range(self.max_iter):
            dists = euclidean_matrix(X_train, self.centroids)
            closest_centroids = np.argmin(dists, axis=1)
            new_centroids = np.array([X_train[closest_centroids == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.allclose(self.centroids, new_centroids, atol=1e-6):
                break
            self.centroids = new_centroids

    def evaluate(self, X):
        dists = euclidean_matrix(X, self.centroids)
        closest_centroids = np.argmin(dists, axis=1)
        return self.centroids[closest_centroids], closest_centroids

# Example usage:
np.random.seed(42)
centers = 3
#X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
#X_train = StandardScaler().fit_transform(X_train)
print("start")
XTrain, YTrain = load_data("ProcessedData", "Training")

print(XTrain)

print("hej")
kmeans = KMeans(n_clusters=centers)
kmeans.fit(XTrain)
class_centers, classification = kmeans.evaluate(XTrain)

# Visualization (still 2D for clarity)
sns.scatterplot(x=XTrain[:, 0], y=XTrain[:, 1], hue=true_labels, style=classification, palette="deep", legend=None)
for centroid in kmeans.centroids:
    plt.plot(centroid[0], centroid[1], 'k+', markersize=10)
plt.show()