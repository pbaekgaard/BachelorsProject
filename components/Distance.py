import numpy as np


def findDistances(centroids, point):
    """Calculate the Euclidean distance from a given point to all centroids."""
    distances = []
    for c in centroids:
        distance = findSingleDistance(c, point)
        distances.append(distance)
    print(f"a random distance: {distances[0]}")
    return distances


def findSingleDistance(centroid, point):
    """Calculate the Euclidean distance"""
    # Calculate the Euclidean distance between two 6d points:
    distance = np.sqrt(np.sum((centroid.xy.reshape(6, -1) - point.xy.reshape(6, -1)) ** 2))
    return distance
