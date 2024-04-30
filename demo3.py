import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Point:
    def __init__(self, xy, label, isIn):
        self.xy = xy
        self.label = label
        self.isIn = isIn


class Cluster:
    def __init__(self, xy, label,radius):
        self.xy = xy
        self.label = label
        self.radius = radius

def initialize_centroids(points, k):
    """Randomly selects k points as initial centroids"""
    selected_indices = np.random.choice(len(points), k, replace=False)
    centroids = []
    for i in selected_indices:
        centroids.append(Cluster(xy=points[i].xy,label= points[i].label, radius=0))
    
    return centroids


def findDistances(centroids, point):
    """Calculate the Euclidean distance from a given point to all centroids."""
    centroids_np = np.array([centroid.xy for centroid in centroids])
    distances = np.sqrt(np.sum((point.xy - centroids_np)**2, axis=1))
    return distances

def findSingleDistance(centroid, point):
    """Calculate the Euclidean distance""" 
    return np.sqrt(np.sum((point.xy - centroid.xy)**2))


def assign_clusters(points, centroids):
    """Assigns each point to the closest centroid"""
    clusters = []
    for point in points:
        distances = findDistances(centroids, point)
        closest_centroid_index = np.argmin(distances)
        clusters.append(closest_centroid_index)
    return clusters

def update_centroids(points, clusters, k):
    """Updates centroids by computing the mean of assigned points and updates using Cluster class"""
    new_centroids = []
    for i in range(k):
        cluster_points = np.array([points[j].xy for j in range(len(points)) if clusters[j] == i])
        if len(cluster_points) > 0:
            new_centroid_xy = np.mean(cluster_points, axis=0)
            # Assume the label of the new centroid to be the same as the label of the first point in the cluster
            centroid_label = points[next(j for j in range(len(points)) if clusters[j] == i)].label
            new_centroids.append(Cluster(xy=new_centroid_xy, label=centroid_label,radius=0))
        else:
            # If no points are in this cluster, we can use the old centroid if necessary:
            # You'll need to decide how to handle this case depending on your application requirements.
            # For now, it just adds a dummy centroid with a default location and label, which should be handled more robustly in production code.
            new_centroids.append(Cluster(xy=np.zeros_like(points[0].xy), label="No Cluster"))
    return new_centroids

def setRadius(centroids,points):
    print("hej")
    for c in centroids:
        longestDistance = 0
        for p in points:
           if(p.label == c.label):
                distance = findSingleDistance(c,p)
                if(distance>longestDistance):
                    longestDistance = distance
                """set the radius"""
        c.radius = longestDistance

def calcInOuts(centroids,points):
    for c in centroids:
        for p in points:
           if(p.label == c.label):
                distance = findSingleDistance(c,p)
                if(distance>c.radius):
                    p.isIn = False




def calcInOutsNewPoint(centroids,point):
    for c in centroids:
        distance = findSingleDistance(c,point)
        shortestDistance = None
        if(distance<c.radius and ((shortestDistance == None ) or distance<shortestDistance  )):
            point.label = c.label
            point.isIn = True
            shortestDistance = distance

    return point
    
        
       
            
            
            
        
        

            


def kmeans(points, k, max_iters=100):
    """Computes k-means clustering using the Cluster class for centroids"""
    centroids = initialize_centroids(points, k)
    print(centroids)  # This will print Cluster objects; you might want to print more informative details

    for _ in range(max_iters):
        clusters = assign_clusters(points, centroids)
        new_centroids = update_centroids(points, clusters, k)

        # Convert centroids to an array for comparison
        old_centroids_array = np.array([centroid.xy for centroid in centroids])
        new_centroids_array = np.array([centroid.xy for centroid in new_centroids])

        # Check if centroids have stopped changing
        if np.allclose(old_centroids_array, new_centroids_array):
            break
        centroids = new_centroids

    return centroids, clusters


# Hardcoded points using the Point class


def findK(points):
    foundKs = []
    for p in points:
        if(p.label not in foundKs):
            foundKs.append(p.label)
    
    return len(foundKs)
        
       
def plotData(centroids, clusters, k, points):
    colors = ['r', 'g']  # Ensure enough colors for the number of clusters, add more if needed

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-3, 20])
    ax.set_ylim([-3, 20])
    ax.set_zlim([-3, 20])
    ax.set_aspect('equal')

    for i in range(k):
        # Collect points assigned to the current cluster
        cluster_points = [points[j] for j in range(len(points)) if clusters[j] == i]
        
        # Plot points in 3D and annotate them with isIn
        for point in cluster_points:
            ax.scatter(point.xy[0], point.xy[1], point.xy[2], s=30, color=colors[i])
            ax.text(point.xy[0], point.xy[1], point.xy[2], f' {point.isIn}', color=colors[i])

        # Plot centroids in 3D
        centroid_coords = centroids[i].xy
        ax.scatter(centroid_coords[0], centroid_coords[1], centroid_coords[2], s=100, color=colors[i], marker='x', edgecolor='k', linewidths=2)

        # Plot sphere around the centroid
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        x = centroid_coords[0] + centroids[i].radius * np.cos(u) * np.sin(v)
        y = centroid_coords[1] + centroids[i].radius * np.sin(u) * np.sin(v)
        z = centroid_coords[2] + centroids[i].radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color=colors[i], alpha=0.1)

    ax.set_title('3D K-Means Clustering with Point and Cluster Classes')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend([f'Cluster {i+1} - {centroids[i].label}' for i in range(k)])
    plt.show()



def main():

    points = [
    Point(xy=np.array([1.5, 1.8, 1.2]), label="A", isIn=True),
    Point(xy=np.array([1.0, 2.0, 1.0]), label="A", isIn=True),
    Point(xy=np.array([6.4, 1.2, 1.3]), label="A", isIn=True),
    Point(xy=np.array([1.2, 1.6, 1.1]), label="A", isIn=True),
    Point(xy=np.array([2.2, 1.9, 1.5]), label="A", isIn=True),
    Point(xy=np.array([6.0, 7.0, 6.5]), label="B", isIn=True),
    Point(xy=np.array([5.2, 6.8, 6.3]), label="B", isIn=True),
    Point(xy=np.array([1.8, 7.2, 6.7]), label="B", isIn=True),
    Point(xy=np.array([1.0, 6.5, 6.9]), label="B", isIn=True),
    Point(xy=np.array([7.2, 7.1, 7.0]), label="B", isIn=True)
]
    
    k = findK(points) # Number of clusters 
    centroids, clusters = kmeans(points, k)
    setRadius(centroids,points)
    calcInOuts(centroids,points)
    newp = Point(xy=np.array([10, 10, 10]), label="C", isIn=False)
    p = calcInOutsNewPoint(centroids,newp)

    clusters.append(0)
    points.append(p)
    plotData(centroids, clusters,k,points)
    


    print("hej")
if __name__ == "__main__":
    main()