from components.Transformer import Transform
from components.Makedataframes import make_dataframes
from components.Objects import Point


def InitData(path: str, WINDOW_SIZE: int):
    """
    Reads the preprocessed data in the given folder
    :param path: str
    :return: transformed time series data
    """

    data, labels, data_test, labels_test = make_dataframes(path, WINDOW_SIZE)
    data = Transform(data)
    points = []
    for idx, dat in enumerate(data):
        point = Point(xy=dat, label=labels[idx])
        points.append(point)
    return points
