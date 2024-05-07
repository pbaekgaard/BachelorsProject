#!/usr/bin/python3
import numpy as np
import pickle
import os
from phases.init import Initialize
from phases.refit import Refit
from components.Objects import Point
from components.Plot import plotData
from components.Makedataframes import make_dataframes
from components.InitData import InitData
import globalvars

WINDOW_SIZE = 1500


def saveModel(centroids, out_points):
    with open("model_data.pkl", "wb") as f:
        pickle.dump((centroids, globalvars.out_points), f)


def loadModel():
    with open("model_data.pkl", "rb") as f:
        _centroids, _points = pickle.load(f)

    return _centroids, _points


def main():
    globalvars.init()

    if not os.path.exists("model_data.pkl"):
        points = InitData("ProcessedData", WINDOW_SIZE)
        centroids = Initialize(points)
        print("centroid radiuses")
        for c in centroids:
            print(f"{c.label}: {c.radius}")
        print("Saving...")
        saveModel(centroids, globalvars.out_points)
        print(f"number of outpoints from inside main before refit: {len(globalvars.out_points)}")
    else:
        print("Loading Existing Model Data...")
        centroids, globalvars.out_points = loadModel()
        print(f"Number of out_points from inside main, after load existing model: {len(globalvars.out_points)}")
        print(f"Number of centroids pre refit: {len(centroids)}")
        newPoints = InitData("NewData", WINDOW_SIZE)
        for currPoint in newPoints:
            centroids = Refit(_centroids=centroids, new_point=currPoint)
        # centroid = Refit(centroids, new_point)
        print(len(globalvars.out_points))
        print(f"Number of centroids after refit: {len(centroids)}")
        print("Saving...")
        saveModel(centroids, globalvars.out_points)

    # 0) Plot new point
    # 1) Implement threshold for amount of new points to create cluster and threshold for single point radius to check for points inside to know if it is within the same cluster
    # 2) Is new point already in a cluster (d <= r -> calcInOut())
    # 3) Save model with new point added to globalvars.out_points[]

    # plotData(centroids, len(centroids), points)  # Visualize the clusters
    print("STATUS REPORT:")
    print(f"Number of centroids: {len(centroids)}")
    for idx, c in enumerate(centroids):
        print(f"Centroid {idx+1}: {c.label}")


if __name__ == "__main__":
    main()
