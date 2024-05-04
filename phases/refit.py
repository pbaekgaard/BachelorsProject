import numpy as np
import glob
from components.Objects import Point, Cluster
from components.Distance import findSingleDistance
TRESH = 1
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

def find_most_centered(close_points):
  """
  Finds the most centered point among the given close points.

  Args:
      close_points: A list of 5 close points.

  Returns:
      The most centered point.
  """

  # Calculate center of mass (excluding current point)
  center_of_mass = np.mean(close_points[:-1], axis=0)

  # Calculate distances to center of mass
  distances = np.linalg.norm(close_points - center_of_mass, axis=1)

  # Find the point with minimum distance
  index = np.argmin(distances)
  return index

def newClusterCreated():
    point_coordinates = [point.xy for point in glob.out_points]
    points_array = np.array(point_coordinates)
    for i, point in enumerate(points_array):
        distances = np.linalg.norm(points_array - point, axis=1)

        close_points = points_array[distances <= TRESH]
        if len(close_points) == 5:
            centerpoint_index = find_most_centered(close_points)
            close_points_indices = np.unique(np.where(np.isin(points_array, close_points))[0])
            print(close_points_indices)
            newCluster = Cluster(xy=glob.out_points[centerpoint_index].xy, label="idk", radius=5)
            for index in close_points_indices[::-1]:
                del glob.out_points[index]
            return True, newCluster
    return False, None

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
    if len(glob.out_points) >= newClusterPointThreshold:
        newClusterWasCreated, newCluster = newClusterCreated()
        if(newClusterWasCreated):
            # Add new clustercenter to centroids array
            centroids.append(newCluster)
            # Rerun refit
            Refit(centroids)
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
