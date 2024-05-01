import os
import numpy as np
from sklearn.cluster import KMeans
import pickle

from classifier import load_data

class DynamicKMeans:
    def __init__(self, n_clusters, model_file=None):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.labels = None
        self.model_file = model_file

        if self.model_file:
            self.load_model()

    def initialize_clusters(self, data, labels):
        unique_labels = np.unique(labels)
        self.n_clusters = len(unique_labels)
        cluster_centers = []
        for label in unique_labels:
            label_data = data[labels == label]
            center = np.mean(label_data, axis=0)
            cluster_centers.append(center)
        self.kmeans = KMeans(n_clusters=self.n_clusters, init=np.array(cluster_centers), n_init=1)
        self.kmeans.fit(data)
        self.labels = self.kmeans.labels_

        if self.model_file:
            self.save_model()

    def add_data_point(self, data_point):
        if self.kmeans is None:
            raise ValueError("Clusters are not initialized yet. Please initialize clusters first.")

        closest_cluster_label = self.kmeans.predict([data_point])[0]
        closest_cluster_center = self.kmeans.cluster_centers_[closest_cluster_label]
        distance = np.linalg.norm(data_point - closest_cluster_center)

        if distance <= self.kmeans.inertia_ / len(self.kmeans.labels_):
            self.kmeans.cluster_centers_[closest_cluster_label] = (
                self.kmeans.cluster_centers_[closest_cluster_label] * len(self.kmeans.labels_[self.kmeans.labels_ == closest_cluster_label]) + data_point
            ) / (len(self.kmeans.labels_[self.kmeans.labels_ == closest_cluster_label]) + 1)
        else:
            # Mark as outlier
            print("Data point marked as outlier.")

        if self.model_file:
            self.save_model()

    def add_new_data_points(self, new_data):
        for data_point in new_data:
            self.add_data_point(data_point)
        self.labels = self.kmeans.predict(new_data)

    def get_clusters(self):
        return self.kmeans.cluster_centers_, self.labels

    def save_model(self):
        if not self.model_file:
            self.model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_kmeans_model.pkl")
        with open(self.model_file, "wb") as f:
            pickle.dump(self, f)

    def load_model(self):
        if not os.path.isfile(self.model_file):
            print("Model file not found. Model will be initialized without loading.")
            return
        with open(self.model_file, "rb") as f:
            loaded_model = pickle.load(f)
            self.__dict__.update(loaded_model.__dict__)

# Example usage:
# Initialize with number of clusters
dynamic_kmeans = DynamicKMeans(n_clusters=3)

# Initialize clusters with initial data and labels
XTest, YTest = load_data("ProcessedData", "Test")

print("XTest: ", XTest, " | Ytest: ", YTest)

initial_data = XTest
initial_labels = YTest
dynamic_kmeans.initialize_clusters(initial_data, initial_labels)

# Add new data
new_data = np.array([[2, 3], [6, 7]])
dynamic_kmeans.add_new_data_points(new_data)

# Get updated clusters and labels
cluster_centers, labels = dynamic_kmeans.get_clusters()
print("Updated cluster centers:", cluster_centers)
print("Updated labels:", labels)