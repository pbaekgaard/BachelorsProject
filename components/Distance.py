import numpy as np


def findDistances(centroids, point):
    """Calculate the Euclidean distance from a given point to all centroids."""
    distances = []
    for c in centroids:
        distance = findSingleDistance(c, point)
        distances.append(distance)
    return distances


def findSingleDistance(centroid, point):
    """Calculate the Euclidean distance"""
    # Calculate the Euclidean distance between two 6d points:
    distance = np.sqrt(np.sum((centroid.xy.reshape(centroid.xy.shape[0]*centroid.xy.shape[1], -1) - point.xy.reshape(centroid.xy.shape[0] * centroid.xy.shape[1], -1)) ** 2))
    return distance


