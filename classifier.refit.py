import os
import numpy as np
import pandas as pd
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from joblib import load

SAMPLELEN = 900

def retrain_classifier(file_path, classifier_path):
    # Load the saved classifier
    classifier = load(classifier_path)

    # Load new data for retraining
    XTrain_new, YTrain_new = load_data(file_path, "Training")

    # Refit the classifier with new data
    classifier.fit(XTrain_new, YTrain_new)

    return classifier
