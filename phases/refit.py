from time import sleep as pause
import numpy as np
import globalvars
from components.Objects import Point, Cluster
from components.Distance import findSingleDistance, findDistances

THRESH = 10
FoundInPoints = []
newClusterPointThreshold = 5
outpointsfails = 0



def CheckIfOutpointsContainsInPoints(centroids):
    global outpointsfails
    globalvars.out_points = globalvars.out_points[::-1]
    for idx, point in enumerate(globalvars.out_points):
        for centroid in centroids:
            distance = findSingleDistance(point, centroid)
            if distance < centroid.radius:
                FoundInPoints.append((point, centroid))
                globalvars.out_points.pop(idx)
                outpointsfails = outpointsfails + 1
                print(f"Found point {point.label} in centroid {centroid.label}")
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
    # oldXy = centroids[centroidIndex].xy
    centroids[centroidIndex].xy = np.mean([centroids[centroidIndex].xy, foundPoint.xy], axis=0)
    # diff = np.linalg.norm(oldXy - centroids[centroidIndex].xy)
    # centroids[centroidIndex].xy = centroids[centroidIndex].xy -  diff/2
    return centroids



def find_most_centered(close_points):
    """
    Finds the most centered point among the given close points.

    Args:
        close_points: A list of 5 close points.
    Args:
        close_points: A list of 5 close points.

    Returns:
        The most centered point.
    """

    # Calculate center of mass (excluding current point)
    center_of_mass = np.mean(close_points[:-1], axis=0)
    # Calculate center of mass (excluding current point)
    center_of_mass = np.mean(close_points[:-1], axis=0)

    # Calculate distances to center of mass
    # distances = np.linalg.norm(close_points - center_of_mass, axis=1)
    tempCentroids = []
    for point in close_points:
        tCent = Point(xy=point)
        tempCentroids.append(tCent)
    distances = findDistances(centroids=tempCentroids, point=Point(xy=center_of_mass))
    # Find the point with minimum distance
    index = np.argmin(distances)
    return index



def newClusterCreated():
    point_coordinates = [point.xy for point in globalvars.out_points]
    points_array = np.array(point_coordinates)
    for i, point in enumerate(points_array):
        # distances2 = np.linalg.norm(points_array - point, axis=1)
        # print(f"Example of distances2: {distances2[3]}")
        distances = findDistances(centroids=globalvars.out_points, point=Point(xy=point))
        distances = np.array(distances)
        close_points = points_array[distances <= THRESH]
        if len(close_points) >= newClusterPointThreshold:
            print(type(close_points))
            print(close_points[1])
            centerpoint_index = find_most_centered(close_points)
            close_points_indices = np.unique(np.where(np.isin(points_array, close_points))[0])
            points_close_to_centroid = []
            for idx in close_points_indices:
                if idx != centerpoint_index:
                    points_close_to_centroid.append(globalvars.out_points[idx])
            radius = np.median(
                findDistances(
                    centroids=points_close_to_centroid, point=Point(xy=globalvars.out_points[centerpoint_index].xy)
                )
            ) / 0.8
            userLabel = input(
                "It looks like you have been doing something new for a while. Please give me a label so i can remember"
                " it for the future: "
            )
            newCluster = Cluster(xy=globalvars.out_points[centerpoint_index].xy, label=userLabel, radius=radius)
            for index in close_points_indices[::-1]:
                del globalvars.out_points[index]
            return True, newCluster
    return False, None




# def newClusterCreated():
#     print(len(globalvars.out_points))
#     for idx, currentPoint in enumerate(globalvars.out_points):
#         # Array of all other points beside the current point
#         otherPoints = globalvars.out_points[:idx] + globalvars.out_points[idx + 1:]
#         # Calculate distances to all other points
#         distances = findDistances(centroids=otherPoints, point=currentPoint)
#         # Find the points that are within the threshold
#         pointsWithinThreshold = []
#         for distance in distances:
#             if distance < THRESH:
#                 pointsWithinThreshold.append(otherPoints[distances.index(distance)])
#         # If there are enough points within the threshold, create a new cluster
#         if len(pointsWithinThreshold) > newClusterPointThreshold:
#             # Get distances from current point to all the close points
#             distancesClose = findDistances(centroids=pointsWithinThreshold, point=currentPoint)
#             # Create the radius from the close points distances
#             radius = np.mean(distancesClose)
#             # Ask the user to label the new cluster
#             userLabel = input("It looks like you have been doing something new for a while. Please give me a label so i can remember")
#             # Create the new cluster
#             newCluster = Cluster(xy=currentPoint.xy, label=userLabel, radius=radius)
#             # Remove the points that are within the threshold from outpoints
#             # And remove the current point from the outpoints.
#             print("Glbalvars outpoints length: ", len(globalvars.out_points))
#             for point in pointsWithinThreshold:
#                 print(f"point in threshold: {point.xy}")
#                 print(f"point in globalvars:{globalvars.out_points[globalvars.out_points.index(point)]}" )
#                 globalvars.out_points.remove(point)
#             globalvars.out_points.remove(currentPoint)
#             print(f"New Cluster Created, outPoints is this length: {len(globalvars.out_points)}")
#             return True, newCluster
#     return False, None




def Refit(_centroids, new_point=None):
    global outpointsfails
    centroids = _centroids
    # Check if there is a new point for the refit
    if new_point is not None:
        globalvars.out_points.append(new_point)
    # Check if there is any points that are in any centroid
    while CheckIfOutpointsContainsInPoints(centroids):
        centroids = Recalibrate(centroids)


    # If a new cluster is created from multiple close outpoints, rerun Refit
    if len(globalvars.out_points) >= newClusterPointThreshold:
        newClusterWasCreated, newCluster = newClusterCreated()
        if newClusterWasCreated:
            # Add new clustercenter to centroids array
            centroids.append(newCluster)
            # Rerun refit
            Refit(centroids)
    print("Outpoints fails: ", outpointsfails)
    return centroids

def Predict(_centroids, point):
    minDistance = 0
    prediction = "Unknown Class"
    for centroid in _centroids:
        distance = findSingleDistance(point, centroid)
        if distance < centroid.radius:
            if minDistance == 0:
                minDistance = distance
                prediction = centroid.label
            elif distance < minDistance:
                minDistance = distance
                prediction = centroid.label
    return prediction

# def checkPointInOutpoints(newPoint):
#     foundOne = False
#     for point in globalvars.out_points:
#         foundOne = checkPointWithNewPoint(point, newPoint)
#         if foundOne:
#             return True

#     return False


# def checkPointWithNewPoint(point, newPoint):
#     for i in range(len(point.xy)):
#         if point.xy[i] != newPoint.xy[i]:
#             return False
#     return True
