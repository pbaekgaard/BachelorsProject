from components.Transformer import Transform
from components.Makedataframes import make_dataframes
from components.Objects import Point
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, PowerTransformer
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
scaler = MinMaxScaler()
normalizer = Normalizer()
from time import sleep as pause


def InitData(path: str, WINDOW_SIZE: int, training: bool = True):
    """
    Reads the preprocessed data in the given folder
    :param path: str
    :return: transformed time series data
    """
    data, labels = [], []
    trainingData, trainingLabels, data_test, labels_test = make_dataframes(path, WINDOW_SIZE)
    if(training):
        data = trainingData
        labels = trainingLabels
    else:
        data = data_test
        labels = labels_test
    

    data = Transform(data)
    # data = scaler.fit_transform(data)
    # data = normalize(data, norm="l2")
    points = []
    columns = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
    blah = 0
    for idx, dat in enumerate(data):

        point = Point(xy=dat, label=labels[idx])
        # if labels[idx] == 'A':
        #     print(f"Point {labels[idx]} has mean: {point.xy}")
        #     # plt.plot(point.xy.T)
        #     blah += 1
        # if labels[idx] == 'B':
        #     print(f"Point {labels[idx]} has mean: {point.xy}")
        #     # plt.plot(point.xy.T)
        #     blah += 1
        # if labels[idx] == 'C':
        #     print(f"Point {labels[idx]} has mean: {point.xy}")
        #     # plt.plot(point.xy.T)
        #     blah += 1
        # if labels[idx] == 'D':
        #     print(f"Point {labels[idx]} has mean: {point.xy}")
        #     plt.plot(point.xy.T)
        #     blah += 1
        #     plt.show()
        #     pause(1500)

        points.append(point)
    return points, labels
