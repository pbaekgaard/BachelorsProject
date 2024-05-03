#!/usr/bin/python3
import numpy as np
import pickle
import os
from phases.init import Initialize
from phases.refit import Refit
from components.Objects import Point
from components.Plot import plotData
import glob
from components.Transformer import transform

def saveModel(centroids, clusters, out_points):
    with open("model_data.pkl", "wb") as f:
        pickle.dump((centroids, clusters, glob.out_points), f)


def loadModel():
    with open("model_data.pkl", "rb") as f:
        _centroids, _clusters, _points = pickle.load(f)

    return _centroids, _clusters, _points


def main():
    glob.init()
    points = [
        Point(xy=np.array([1.5, 1.8, 1.2]), label="A", isIn=True),
        Point(xy=np.array([1.0, 2.0, 1.0]), label="A", isIn=True),
        Point(xy=np.array([6.4, 1.2, 1.3]), label="A", isIn=True),
        Point(xy=np.array([1.2, 1.6, 1.1]), label="A", isIn=True),
        Point(xy=np.array([2.2, 1.9, 1.5]), label="A", isIn=True),
        Point(xy=np.array([6.0, 7.0, 6.5]), label="B", isIn=True),
        Point(xy=np.array([5.2, 6.8, 6.3]), label="B", isIn=True),
        Point(xy=np.array([1.8, 7.2, 6.7]), label="B", isIn=True),
        Point(xy=np.array([1.0, 6.5, 6.9]), label="B", isIn=True),
        Point(xy=np.array([7.2, 7.1, 7.0]), label="B", isIn=True),
    ]

    new_point = Point(xy=np.array([1, 1.9, 1.1]))
    new_point2 = Point(xy=np.array([2, 1.9, 1.1]))
    new_point3 = Point(xy=np.array([100, 1.9, 1.1]))
    new_point4 = Point(xy=np.array([101, 1.9, 1.1]))
    new_point5 = Point(xy=np.array([100.5, 1.9, 1.1]))
    new_point6 = Point(xy=np.array([100.6, 1.9, 1.1]))
    new_point7 = Point(xy=np.array([100.3, 1.9, 1.1]))
    newPoints = [new_point, new_point2, new_point3, new_point4, new_point5, new_point6, new_point7]
    if not os.path.exists("model_data.pkl"):
        k, centroids, clusters = Initialize(points)

        print("Saving...")
        saveModel(centroids, clusters, glob.out_points)
        print(f"number of outpoints from inside main before refit: {len(glob.out_points)}")
    else:
        print("Loading...")
        centroids, clusters, glob.out_points = loadModel()
        k = len(centroids)
        print(f"Number of out_points from inside main, after load existing model: {len(glob.out_points)}")
        for currPoint in newPoints:
            centroid = Refit(_centroids=centroids, new_point=currPoint)
        # centroid = Refit(centroids, new_point)
        print(len(glob.out_points))
        print("Saving...")
        saveModel(centroids, clusters, glob.out_points)

    # 0) Plot new point
    # 1) Implement threshold for amount of new points to create cluster and threshold for single point radius to check for points inside to know if it is within the same cluster
    # 2) Is new point already in a cluster (d <= r -> calcInOut())
    # 3) Save model with new point added to glob.out_points[]

    plotData(centroids, clusters, k, points)  # Visualize the clusters


if __name__ == "__main__":
    main()
