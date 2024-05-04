import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plotData(centroids, clusters, k, points):
    cluster_colors = ["r", "g", "m"]  # Cluster colors
    unassigned_color = "b"  # Color for unassigned points

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-3, 20])
    ax.set_ylim([-3, 20])
    ax.set_zlim([-3, 20])
    ax.set_aspect("equal")

    # Plot unassigned points first
    unassigned_points = [points[j] for j in range(len(points)) if clusters[j] == -1]
    for point in unassigned_points:
        ax.scatter(point.xy[0], point.xy[1], point.xy[2], s=30, color=unassigned_color)
        ax.text(point.xy[0], point.xy[1], point.xy[2], f" {point.isIn}", color=unassigned_color)

    for i in range(k):
        # Collect points assigned to the current cluster
        cluster_points = [points[j] for j in range(len(points)) if clusters[j] == i]

        # Plot points in 3D and annotate them with isIn
        # for point in cluster_points:
        #     ax.scatter(point.xy[0], point.xy[1], point.xy[2], s=30, color=cluster_colors[i])
        #     ax.text(point.xy[0], point.xy[1], point.xy[2], f" {point.isIn}", color=cluster_colors[i])

        # Plot centroids in 3D
        centroid_coords = centroids[i].xy
        ax.scatter(
            centroid_coords[0],
            centroid_coords[1],
            centroid_coords[2],
            s=100,
            color=cluster_colors[i],
            marker="o",
            edgecolor="k",
            linewidths=2,
        )

        # Plot sphere around the centroid
        u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 50j]
        x = centroid_coords[0] + centroids[i].radius * np.cos(u) * np.sin(v)
        y = centroid_coords[1] + centroids[i].radius * np.sin(u) * np.sin(v)
        z = centroid_coords[2] + centroids[i].radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color=cluster_colors[i], alpha=0.1)

    ax.set_title("3D K-Means Clustering with Point and Cluster Classes")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.legend([f"Cluster {i+1} - {centroids[i].label}" for i in range(k)])
    plt.show()
