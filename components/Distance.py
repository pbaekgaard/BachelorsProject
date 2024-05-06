import numpy as np


def findDistances(centroids, point):
    """Calculate the Euclidean distance from a given point to all centroids."""
    centroids_np = np.array([centroid.xy for centroid in centroids])
    distances = np.sqrt(np.sum((point.xy - centroids_np) ** 2, axis=1))
    return distances


def findSingleDistance(centroid, point):
    """Calculate the Euclidean distance"""
    distance = np.sqrt(np.sum((point.xy - centroid.xy) ** 2))
    return distance
