from components.Transformer import Transform
from components.Makedataframes import make_dataframes
from components.Objects import Point
from sklearn.preprocessing import MinMaxScaler, Normalizer
scaler = MinMaxScaler()
normalizer = Normalizer()


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
    points = []
    for idx, dat in enumerate(data):
        point = Point(xy=dat, label=labels[idx])
        points.append(point)


    return points, labels
