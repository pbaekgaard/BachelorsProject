#!/usr/bin/python3
import numpy as np
import pickle
import os
from phases.init import Initialize
from phases.refit import Refit, Predict
from components.Objects import Point
from components.Plot import plotData
from components.Makedataframes import make_dataframes
from components.InitData import InitData
from phases.fitinit import InitFit
import globalvars
from sklearn.metrics import precision_score, recall_score, accuracy_score
from components.Distance import findSingleDistance

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
    refit = True
    if not os.path.exists("model_data.pkl"):
        points,labels = InitData("ProcessedData", WINDOW_SIZE, training=True)
        centroids = Initialize(points)
        print("centroid radiuses")
        for c in centroids:
            print(f"{c.label}: {c.radius}")
        print("Saving...")
        saveModel(centroids, globalvars.out_points)
        print(f"number of outpoints from inside main before refit: {len(globalvars.out_points)}")
        # calculate the distance (using findDistance) between the centroids, print if the distance is smaller than radius + radius
        for idx, c in enumerate(centroids):
            for idx2, c2 in enumerate(centroids):
                if idx != idx2:
                    distance = findSingleDistance(c, c2)
                    # print(f"Distance in main: {distance}")
                    if distance < (c.radius + c2.radius):
                        print(f"Distance between {c.label} and {c2.label} is {distance} which is smaller than {c.radius + c2.radius}")
                        print(f"Point {c.label} has point: {c.xy}")
                        print(f"Point {c2.label} has point: {c2.xy}")


    else:
        print("Loading Existing Model Data...")
        centroids, globalvars.out_points = loadModel()
        print(f"Number of out_points from inside main, after load existing model: {len(globalvars.out_points)}")
        print(f"Number of centroids pre refit: {len(centroids)}")
        newPoints, labels = InitData("NewData", WINDOW_SIZE, training=True)
        testPoints, testLabels = InitData("StupidTestData", WINDOW_SIZE, training=True)
        predictions = []
        actual = testLabels
        if(refit):
            for currPoint in newPoints:
                centroids = Refit(_centroids=centroids, new_point=currPoint)
        for testPoint in testPoints:
            prediction = Predict(centroids, testPoint)
            predictions.append(prediction)
        print(f"Predicted labels: {predictions}")
        print(f"Actual labels: {actual}")
        # Calculate precision
        precision = precision_score(actual, predictions, average='weighted')

        # Calculate accuracy
        accuracy = accuracy_score(actual, predictions)
        # Calculate precision
        precision_micro = precision_score(actual, predictions, average='micro')
        precision_macro = precision_score(actual, predictions, average='macro')

        # Calculate recall
        recall_micro = recall_score(actual, predictions, average='micro')
        recall_macro = recall_score(actual, predictions, average='macro')

        # PRINT METRICS
        print(f"Accuracy: {accuracy}")
        print(f"Precision (Micro): {precision_micro}")
        print(f"Recall (Micro): {recall_micro}")
        print(f"Precision (Macro): {precision_macro}")
        print(f"Recall (Macro): {recall_macro}")


        # print(len(globalvars.out_points))
        # print(f"Number of centroids after refit: {len(centroids)}")
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
        print(f"Radius: {c.radius}")
        print(f"Points: {c.xy}\n")
        for idx2, c2 in enumerate(centroids):
            if idx != idx2:
                distance = findSingleDistance(c, c2)
                print(f"Distance between {c.label} and {c2.label} is {distance}. They have a combined radius of: {c.radius + c2.radius}")


if __name__ == "__main__":
    main()
