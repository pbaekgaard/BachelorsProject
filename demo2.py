#! /usr/bin/python3
import numpy as np
from pandas.io.formats.format import math

class pointClass:
    def __init__(self, point, name, cluster = None, distance = 0):
        self.point = point
        self.cluster = cluster
        self.distanceToCenter : float = distance
        self.name = name


class clusterClass:
    def __init__(self, name, clusterCenter, clusterPoints = None):
        self.name = name
        self.clusterCenter = clusterCenter
        self.clusterPoints = clusterPoints if clusterPoints is not None else []


# Define the matrix A
A = np.array([2, 5])
B = np.array([4, 10])
C = np.array([3, 10])
D = np.array([3, 4])
E = np.array([2, 3])
F = np.array([4, 4])
G = np.array([3, 20])
H = np.array([6, 4])
I = np.array([30, 70])

point1 = pointClass(A,"Point1")
point2 = pointClass(B,"Point2")
point3 = pointClass(C,"Point3")
point4 = pointClass(D,"Point4")
point5 = pointClass(E,"Point5")
point6 = pointClass(F,"Point6")
point7 = pointClass(G,"Point7")
point8 = pointClass(H,"Point8")
point9 = pointClass(I,"Point9")



points = [point3,point4,point5,point6,point7,point8, point9]

clusters = []
iteration = 0

def distance(v1, v2):
    diff = v1 - v2
    square_diff = np.power(diff, 2)
    return np.sqrt(np.sum(square_diff))

def overThresholdForNewCalc(newCenter, oldCenter):
    if not newCenter.shape == oldCenter.shape:
        raise ValueError("Arrays must have the same shape")

    # Calculate absolute difference element-wise
    abs_diff = np.abs(newCenter - oldCenter)

    # Calculate relative difference as a percentage
    relative_diff = abs_diff / np.maximum(np.abs(newCenter), np.abs(oldCenter)) * 100

    # Check if all elements have a difference less than 5%
    return np.any(relative_diff > 5)

def recalculateClusterCenters():
    global clusters
    global iteration
    thresholdFlag : bool = False
    for c in clusters:
        x = [p.point[0] for p in c.clusterPoints]
        y = [p.point[1] for p in c.clusterPoints]
        if(len(c.clusterPoints) != 0):
            newCenter = pointClass(point=np.array([sum(x)/len(c.clusterPoints), sum(y)/len(c.clusterPoints)]), name="newCenter")
            if(overThresholdForNewCalc(newCenter.point, c.clusterCenter.point)):
                c.clusterCenter = newCenter
                thresholdFlag = True
    iteration = iteration + 1
    if thresholdFlag:
        calculatePointsClusterAssociation()



def calculatePointsClusterAssociation():
    global clusters
    global points
    print(f"ITERATION {iteration}")
    for c in clusters:
        print(f"cluster {c.name} with points:")
        for point in c.clusterPoints:
            print(f"Point: {point.point}, Cluster: {point.cluster.name if not None else 'None'}")
    print("\n\n\n")
    for p in points:
        pointSwitchedCluster : bool = False
        pointDistance = math.inf 
        currentChosenCluster = None
        for c in clusters:
            distanceToCluster = distance(p.point, c.clusterCenter.point)
            if(distanceToCluster < pointDistance):
                pointDistance = distanceToCluster
                currentChosenCluster = c
        if(currentChosenCluster is not p.cluster):
            pointSwitchedCluster = True
            print("CLUSTER CHANGE!")

        if(p.cluster is not None):
            print("\n\n\n CURRENT POINT IN OPERATION:")
            print(f"point {p.point} with current cluster: {p.cluster.name}")
        newPoint = p
        print(pointDistance)
        print(currentChosenCluster.name)
        newPoint.distanceToCenter = pointDistance
        newPoint.cluster = currentChosenCluster
        if(p.cluster is not None):
            print("\n\n\n CURRENT POINT IN OPERATION:")
            print(f"point {p.point} with current cluster: {p.cluster.name}")
            print(f"newpoint {newPoint.point} with current cluster: {p.cluster.name}")
        for cluster in clusters:
            if(currentChosenCluster is not None):
                if(cluster.name == p.cluster.name and pointSwitchedCluster):
                    print("REMOVING OLD POINT")
                    for idx, n in enumerate(cluster.clusterPoints):
                        print(f"Checking point with x: {n.point[0]}, y: {n.point[1]} with point with x: {p.point[0]}, y: {p.point[1]}")
                        if n.point[0] == p.point[0] and n.point[1] == p.point[1]:
                            print("found point to remove")
                if(currentChosenCluster.name == cluster.name):
                    foundOne : bool = False
                    for idx, n in enumerate(cluster.clusterPoints):
                        if n == p:
                            cluster.clusterPoints[idx] = newPoint
                            foundOne = True
                    if not foundOne:
                        cluster.clusterPoints.append(p)
    print(f"END OF ITERATION {iteration}")
    for c in clusters:
        print(f"cluster {c.name} with points:")
        for point in c.clusterPoints:
            print(f"Point: {point.point}, Cluster: {point.cluster.name if not None else 'None'}")
    print("\n\n\n")

    recalculateClusterCenters()


def main():
    global clusters
    newClusterCenterPoint = pointClass(point=A, name="myFirstCenter")
    newClusterCenterPoint2 = pointClass(point=B, name="mySecondCenter")
    cluster1 = clusterClass(clusterCenter=newClusterCenterPoint, name="Cluster 1")
    cluster2 = clusterClass(clusterCenter=newClusterCenterPoint2, name="Cluster 2")
    clusters.append(cluster1)
    clusters.append(cluster2)
    calculatePointsClusterAssociation()

if __name__ == "__main__":
    main()









  

 





