import numpy as np
import random 
import pandas as pd

"""
def k_means_cluster(k, points):
  # Initialization: choose k centroids (Forgy, Random Partition, etc.)
  centroids = [c1, c2, ..., ck]
  
  # Initialize clusters list
  clusters = [[] for _ in range(k)]
  
  # Loop until convergence
  converged = false
  while not converged:
      # Clear previous clusters
      clusters = [[] for _ in range(k)]
  
      # Assign each point to the "closest" centroid 
      for point in points:
          distances_to_each_centroid = [distance(point, centroid) for centroid in centroids]
          cluster_assignment = argmin(distances_to_each_centroid)
          clusters[cluster_assignment].append(point)
      
      # Calculate new centroids
      #   (the standard implementation uses the mean of all points in a
      #     cluster to determine the new centroid)
      new_centroids = [calculate_centroid(cluster) for cluster in clusters]
      
      converged = (new_centroids == centroids)
      centroids = new_centroids
      
      if converged:
          return clusters
"""

def km(data):
    K = 9
    centroids_indices = random.sample(range(len(data)), K) 
    centroids = [data[index] for index in centroids_indices]
    
    clusters = [[] for _ in range(K)]

    converged = False
    while not converged:
        clusters = [[] for _ in range(K)]

        for point in data:
          distances_to_each_centroid = [euclidean(point, centroid) for centroid in centroids]
          cluster_assignment = np.argmin(distances_to_each_centroid)
          clusters[cluster_assignment].append(point)

        new_centroids = [calculate_centroid(cluster) for cluster in clusters]
        
        converged = all(np.array_equal(new, old) for new, old in zip(new_centroids, centroids))
        centroids = new_centroids

        if converged:
          return clusters
        
def euclidean(a, b):
    """
    Returns Euclidean distance between vectors a and b
    """
    return np.sqrt(np.sum((np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64))**2))

def calculate_centroid(cluster):
  return np.mean(cluster, axis=0)

def read_data(filename):
    dataset = []
    with open(filename, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i + 1])
            dataset.append([label, attribs])
    return dataset

def inplace_min_max_scaling(data):
    maximum = 255.0

    for i, row in enumerate(data):
        row = np.array(row[1], dtype=np.float64)
        scaled = row / maximum
        data[i][1] = scaled


if __name__ == '__main__':
    data = read_data('mnist_train.csv')
    inplace_min_max_scaling(data)

    features = [row[1] for row in data]
    clusters = km(features)

    data_representation = []
    for i, cluster in enumerate(clusters):
      label = f"class-{i}"
      for image in cluster:
        data_representation.append([label, image])
    print(pd.DataFrame(data_representation, columns=["label", "features"]))
