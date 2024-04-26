from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class pointClass:
  def __init__(self, point, cluster, name):
    self.point = point
    self.cluster = cluster
    self.name = name


class clusterClass:
  def __init__(self, clusterName, clusterPoint):
    self.clusterName = clusterName
    self.clusterPoint = clusterPoint


# Define the matrix A
A = np.array([2, 5])
B = np.array([1, 4])
C = np.array([3, 10])
D = np.array([3, 4])
E = np.array([2, 3])
F = np.array([4, 4])
G = np.array([3, 20])
H = np.array([6, 4])


#C = np.array([[1, 4, 5, 12], 
 #             [-5, 8, 9, 0],
  #            [-6, 7, 11, 19]])




Cluster1 = np.array([4, 13])

Cluster2 = np.array([2, 10])

#Cluster2 = np.array([[1, 30, 5, 12], 
      #        [50, 80, 5, 100],
     #         [60, 2, 300, 300]])






Cluster11 = clusterClass("C1",Cluster1)
Cluster22 = clusterClass("C2",Cluster2)




point1 = pointClass(A, Cluster11,"Point1")
point2 = pointClass(B, Cluster22,"Point2")
point3 = pointClass(C, Cluster11,"Point3")
point4 = pointClass(D, Cluster22,"Point4")
point5 = pointClass(E, Cluster11,"Point5")
point6 = pointClass(F, Cluster22,"Point6")
point7 = pointClass(G, Cluster11,"Point7")
point8 = pointClass(H, Cluster22,"Point8")




points = [point1,point2,point3,point4,point5,point6,point7,point8]

clusters = [Cluster11,Cluster22]


# Plotting
plt.figure(figsize=(8, 8))

# Plot clusters
for cluster in clusters:
    plt.scatter(cluster.clusterPoint[0], cluster.clusterPoint[1], s=150, marker='s', label=f"{cluster.clusterName} (cluster)")

# Plot points
for pt in points:
    plt.scatter(pt.point[0], pt.point[1], label=pt.name if pt.cluster not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(pt.point[0], pt.point[1], f' {pt.name}', fontsize=12, verticalalignment='bottom')

# Set plot features
plt.title('Plot of Points and Clusters')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.grid(True)
plt.show()


def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))  

def distanceToCluster():
    storedDistance = 10000000000
    for p in points:
        for c in clusters:
            d = distance(c.clusterPoint,p.point)
         #   print("distance= "+str(d))
            #print("stored= "+str(storedDistance))
            if(d<=storedDistance):
              
                p.cluster = c
                storedDistance = d
        storedDistance = 1000000000
    recalc()
           

#recompute the centroid for each cluster


def recalc():
    global clusters
    global points
    newclusters = []

    for c in clusters:
        pointsInCluster = []
        for p in points:
            if (distance(p.cluster.clusterPoint, c.clusterPoint) == 0):
                    pointsInCluster.append(p.point)
      
        if(len(pointsInCluster)>0):
            newcluster = np.sum(pointsInCluster, axis=0) / len(pointsInCluster)
            pointsInCluster.clear()
            
            clusterNew = clusterClass(c.clusterName,newcluster)
            newclusters.append(clusterNew)
          
            if(distance(clusterNew.clusterPoint,c.clusterPoint) > 0):
                clusters = newclusters
                #distance recalc

                distanceToCluster()
            else:
                print("tomt")






#run recalc

for p in points:
    print("--------")
    print(p.name)
    print(p.cluster.clusterName)
    print("--------")

distanceToCluster()

print("1 iteration")
#1 iteration
for p in points:
    print("--------")
    print(p.name)
    print(p.cluster.clusterName)
    print("--------")

          
   









  

 





