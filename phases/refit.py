import numpy as np
import glob
from components.Objects import Point
from components.Distance import findSingleDistance

FoundInPoints = []
newClusterPointThreshold = 5

def CheckIfOutpointsContainsInPoints(centroids):
    glob.out_points = glob.out_points[::-1]
    for idx, point in enumerate(glob.out_points):
        for centroid in centroids:
            distance = findSingleDistance(point, centroid)
            if distance < centroid.radius:
                FoundInPoints.append((point, centroid))
                glob.out_points.pop(idx)
                return True
    return False


def Recalibrate(_centroids):
    centroids = _centroids
    foundPoint, centroid = FoundInPoints[0]
    FoundInPoints.pop(0)
    centroidIndex = None

    # Find the index for centroid to recalibrate
    for index, centroidFromArr in enumerate(centroids):
        if centroid.label == centroidFromArr.label:
            centroidIndex = index
            break

    # Recalibrate centroid
    centroids[centroidIndex].xy = np.mean([centroids[centroidIndex].xy, foundPoint.xy], axis=0)
    return centroids

def newClusterCreated():
    return False

def Refit(_centroids, new_point=None):
    centroids = _centroids
    # Check if there is a new point for the refit
    
    if new_point is not None and checkPointInOutpoints(new_point) is False:
        print(f"Currently working with point: {new_point.xy}")
        glob.out_points.append(new_point)
    # Check if there is any points that are in any centroid
    while CheckIfOutpointsContainsInPoints(centroids):
        centroids = Recalibrate(centroids)
    
    # If a new cluster is created from multiple close outpoints, rerun Refit
    if len(glob.out_points) >= newClusterPointThreshold and newClusterCreated():
        pass
    return centroids

def checkPointInOutpoints(newPoint):
    foundOne = False
    for point in glob.out_points:
        foundOne = checkPointWithNewPoint(point, newPoint)
        if foundOne:
            return True
    
    return False

def checkPointWithNewPoint(point, newPoint):
    for i in range(len(point.xy)):
        if point.xy[i] != newPoint.xy[i]:
            return False
    return True