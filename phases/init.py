from itertools import groupby
from components.Objects import Cluster, Point
from components.Distance import findDistances, findSingleDistance
import numpy as np
import globalvars

RAD = 10


def initialize_centroids(points, k):
    """Randomly selects one point from each label as initial centroids"""
    unique_labels = np.unique([point.label for point in points])
    centroids = []

    # Select one point from each label group
    for label in unique_labels:
        label_points = [point for point in points if point.label == label]
        if not label_points:
            raise ValueError(f"No points with label {label}")

        selected_point = np.random.choice(label_points)
        centroids.append(Cluster(xy=selected_point.xy, label=selected_point.label, radius=RAD))

    # If k is greater than the number of unique labels, raise a ValueError
    if k > len(centroids):
        raise ValueError("Number of centroids (k) cannot be greater than the number of unique labels")

    print("these are the selected centroids")
    for c in centroids:
        print(c.label)
    return centroids[:k]  # Return only k centroids if more than k centroids were selected


def assign_clusters(points, centroids):
    """Assigns each point to the closest centroid"""
    clusters = []
    for point in points:
        distances = findDistances(centroids, point)
        print(distances[0])
        closest_centroid_index = np.argmin(distances)
        clusters.append(closest_centroid_index)
    return clusters


def update_centroids(points, clusters, k):
    """Updates centroids by computing the mean of assigned points and updates using Cluster class"""
    new_centroids = []
    for i in range(k):
        cluster_points = []
        cluster_labels = []
        for j in range(len(points)):
            if clusters[j] == i:
                cluster_points.append(points[j].xy)
                cluster_labels.append(points[j].label)
        cluster_points = np.array(cluster_points)
        cluster_labels = np.array(cluster_labels)
        if len(cluster_points) > 0:
            new_centroid_xy = np.mean(cluster_points, axis=0)
            centroid_label = cluster_labels[0]
            new_centroids.append(Cluster(xy=new_centroid_xy, label=centroid_label, radius=RAD))
        else:
            new_centroids.append(Cluster(xy=np.zeros_like(points[0].xy), label="No Cluster", radius=RAD))
    return new_centroids


def setRadius(centroids, points):
    for c in centroids:
        longestDistance = 0
        for p in points:
            if p.label == c.label:
                distance = findSingleDistance(c, p)
                print(f"Point shape: {p.xy.shape}")
                print(f"Distance: {distance}")
                print(f"Longest Distance: {longestDistance}")
                if distance > longestDistance:
                    longestDistance = distance
                """set the radius"""
        c.radius = longestDistance


def calcInOuts(centroids, points):
    for c in centroids:
        for p in points:
            if p.label == c.label:
                distance = findSingleDistance(c, p)
                if distance > c.radius:
                    p.isIn = False


def calcInOutsNewPoint(centroids, point):
    for c in centroids:
        distance = findSingleDistance(c, point)
        shortestDistance = None
        if distance < c.radius and ((shortestDistance == None) or distance < shortestDistance):
            point.label = c.label
            point.isIn = True
            shortestDistance = distance

    return point


def kmeans(points, k, centroids, clusters, max_iters=100):
    """Computes k-means clustering using the Cluster class for centroids"""
    centroids = initialize_centroids(points, k)
    # print(centroids)  # This will print Cluster objects;

    for _ in range(max_iters):
        clusters = assign_clusters(points, centroids)
        new_centroids = update_centroids(points, clusters, k)

        # Convert centroids to an array for comparison
        old_centroids_array = np.array([centroid.xy for centroid in centroids])
        new_centroids_array = np.array([centroid.xy for centroid in new_centroids])
        print("Old Centroids: ")
        for c in centroids:
            print(c.label)
        print("New Centroids: ")
        for c in new_centroids:
            print(c.label)

        # Check if centroids have stopped changing
        if np.allclose(old_centroids_array, new_centroids_array):
            break
        centroids = new_centroids

    return centroids, clusters


# Main calls


def findK(points):
    foundKs = []
    for p in points:
        if p.label not in foundKs:
            foundKs.append(p.label)

    return len(foundKs)


def initSetup(points, k):
    centroids, clusters = kmeans(points, k, [], [])
    print(k)
    setRadius(centroids, points)
    calcInOuts(centroids, points)

    for point in points:
        if point.isIn == False:
            globalvars.out_points.append(point)

    return centroids, clusters


def Initialize(points):
    centroids = []
    points.sort(key=lambda point: point.label)
    for label, group in groupby(points, key=lambda point: point.label):
        group = list(group)
        centroid_point = np.mean([point.xy for point in group], axis=0)
        print(f"Centroid Shape: {centroid_point.shape}")
        testDistance = findSingleDistance(Point(xy=np.array([[1,1,1],[1,0,1]])), Point(xy=np.array([[1,1,1],[1,1,1]])))
        print(f"Test Distance: {testDistance}")
        # Calculate the radius of the cluster by finding the maximum distance from the centroid to any point in the cluster
        distancesFromCentroid = findDistances(centroids=group, point=Point(xy=centroid_point, label=label))
        radius = np.min(distancesFromCentroid, axis=0)
        group = list(group)
        centroid = Cluster(xy=centroid_point, label=label, radius=radius)
        centroids.append(centroid)
    return centroids


# def Initialize(points):
#     k = findK(points)  # Number of clusters/labels
#     centroids, clusters = initSetup(points, k)
#     return k, centroids, clusters
