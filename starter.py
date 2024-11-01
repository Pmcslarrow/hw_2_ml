import numpy as np 
from collections import Counter
import heapq
from sklearn import metrics
import pandas as pd

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.min_distance = float('inf')
#         self.class_label = None

#     def get_data(self):
#         return self.data
    
#     def get_min_distance(self):
#         return self.min_distance
    
#     def get_class(self):
#         return self.class_label

#     def set_class(self, string):
#         """
#         Parameters
#         ----------
#         string: The string of the new class you want to assign
#         """
#         self.class_label = string

#     def set_min_distance(self, new_min):
#         """
#         Parameters
#         ----------
#         new_min: The new lowest distance to this point (float)
#         """
#         self.min_distance = new_min


def accuracy(actuals, predicted):
    correct = 0
    for i in range(len(actuals)):
        if actuals[i] == predicted[i]:
            correct += 1
    return correct / len(actuals)

def conf_matrix(actuals, predicted):
    print('        0  1  2  3  4  5  6  7  8  9')
    return metrics.confusion_matrix(actuals, predicted)

def inplace_min_max_scaling(data):
    maximum = 255.0

    for i, row in enumerate(data):
        row = np.array(row[1], dtype=np.float64)
        scaled = row / maximum
        data[i][1] = scaled

def pearson_correlation(a, b):
    """
    Returns Pearson correlation distance between vectors a and b
    """
    if len(a) != len(b):
        raise ValueError("Arrays must be of the same length")
    
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    mean_a = np.mean(a)
    mean_b = np.mean(b)

    numerator = (np.sum((a - mean_a) * (b - mean_b)))
    denominator = (np.sqrt(np.sum((a - mean_a) ** 2)) * np.sqrt(np.sum((b - mean_b) ** 2)))

    if denominator == 0.0:
        return 0.0
        
    return 1 - (numerator / denominator)

def hamming(a, b):
    """
    Returns Hamming distance between vectors a and b
    """
    if len(a) != len(b):
        raise ValueError("Arrays must be of the same length")
    
    return sum(pix1 != pix2 for pix1, pix2 in zip(a, b))

def euclidean(a, b):
    """
    Returns Euclidean distance between vectors a and b
    """
    return np.sqrt(np.sum((np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64))**2))

def cosim(a, b):
    """
    Returns Cosine Similarity between vectors a and b.
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    dot_product = np.dot(a, b)
    return 1.0 - (dot_product / (norm_a * norm_b))

def get_k_sorted_distances(test_row, train, metric='euclidean', k=3):
    """
    For a single test row, this iterates through
    every single training example and calculates the 
    distance between them, and stores it in the distances 
    list, sorted. 
    """
    distances = []
    for train_row in train:
        if metric == 'euclidean':
            calc = euclidean(test_row[1], train_row[1])
        elif metric == 'cosim':
            calc = cosim(test_row[1], train_row[1])
        elif metric == 'hamming':
           calc = hamming(test_row[1], train_row[1])
        elif metric == 'pearson_correlation':
           calc = pearson_correlation(test_row[1], train_row[1])
        else:
           raise ValueError(f'metric \'{metric}\' is not a valid option')
        distances.append([calc, train_row[0]])
    return heapq.nsmallest(k, distances)

def get_most_common_label(distances):
    """
    Parameters
    ----------
    distances: A sorted array containing the closest distances and labels associated. k smallest

    Returns
    -------
    The most common label from the k nearest neighbors
    """
    label_counter = Counter(label for _, label in distances)
    return label_counter.most_common(1)[0][0]

def knn(train, query, metric, k=4):
    """
    Parameters
    ----------
    train: The training dataset
    query: The test data you want to predict
    metric: String of which distance measurement you want to use i.e. 'euclidean'

    Returns
    -------
    Returns one array of true labels and one array of the predicted labels 
    """

    print(f"     Running KNN for k = {k} with {metric} metric")
    actuals = []
    predictions = []
    for test_example in query:
        k_distances = get_k_sorted_distances(test_example, train, metric=metric, k=k)
        most_common_label = get_most_common_label(k_distances)
        actuals.append(test_example[0])
        predictions.append(most_common_label)
    return actuals, predictions

def get_random_centroids_michael(data, k):
    data_copy = data.copy()
    np.random.shuffle(data_copy)
    num_points = len(data_copy)
    chunk_size = int(np.ceil(num_points / k))

    centroids = [
        np.mean(data_copy[idx: idx + chunk_size], axis=0) if idx + chunk_size <= num_points 
        else np.mean(data_copy[idx:], axis=0)
        for idx in range(0, num_points, chunk_size)
    ]

    return centroids

def kmeans(data, metric, k=10):
    centroids = get_random_centroids_michael(data, k)

    clusters = None
    cluster_memberships = None

    converged = False
    while not converged:
        clusters = [[] for _ in range(k)]
        cluster_memberships = []

        for point in data:
            if metric == 'euclidean':
                point_to_centroid_dists = [euclidean(point, centroid)
                                        for centroid in centroids]
            elif metric == 'cosim':
                point_to_centroid_dists = [cosim(point, centroid)
                                        for centroid in centroids]
            elif metric == 'hamming':
                point_to_centroid_dists = [hamming(point, centroid)
                                        for centroid in centroids]
            elif metric == 'pearson_correlation':
                point_to_centroid_dists = [pearson_correlation(point, centroid)
                                        for centroid in centroids]
            else:
                raise ValueError(f'metric \'{metric}\' is not a valid option')
            cluster_membership = np.argmin(point_to_centroid_dists)
            clusters[cluster_membership].append(point)
            cluster_memberships.append(cluster_membership)
        
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]

        converged = all([np.array_equal(centroid, new_centroid) for centroid, new_centroid in zip(centroids, new_centroids)])
        centroids = new_centroids

    return cluster_memberships
   
# def kmeans(train, query, metric):
#     num_clusters = 9
#     has_converged = False
#     last_iteration_centroids = None
#     next_iteration_centroids = None
    
#     while not has_converged:
#       nodes = initialize_nodes(train)  # Converts each row to a node representation
#       if not next_iteration_centroids:
#         centroids = select_random_centroids(nodes, num_clusters) # Gets the k random mean clusters from the training set
#         last_iteration_centroids = centroids
#       else:
#         centroids = next_iteration_centroids
#         last_iteration_centroids = centroids
#       assign_labels_to_nodes(nodes, centroids, metric) # Assigns a class name to the node of the closest mean cluster
#       updated_data = generate_dataset_from_labeled_nodes(nodes) # Creates a new dataset of label data and flattened representations    {class-8 ,   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...}
#       next_iteration_centroids = get_means_from_labeled_data(updated_data)
      
#       if last_iteration_centroids is not None and calculate_converge(last_iteration_centroids, next_iteration_centroids):
#         has_converged = True
#     return updated_data

def calculate_converge(last_centroids, next_centroids):
  epsilon = 1e-4
  for i in range(len(last_centroids)):
    difference = np.linalg.norm(np.array(last_centroids[i].get_data()) - np.array(next_centroids[i].get_data()))
    #print(last_centroids[i].get_class(), difference)
    if difference >= epsilon:
      return False
  return True

# def initialize_nodes(train):
#     new_dataset = []
#     for row in train:
#         new_dataset.append(Node(row))
#     return np.array(new_dataset)

def select_random_centroids(train, k):
    k_random_examples = np.random.choice(train, size=k)
    for i in range(len(k_random_examples)):
      k_random_examples[i].set_class('class-' + str(i))  # Assigns a unique class to each starting mean point
    return k_random_examples

def assign_labels_to_nodes(train, k_random_examples, metric):
    """
    Calculates the distance from each training example to the
    k random examples and assigns the class label of the nearest 
    random example to each training example.
    """
    for class_node in k_random_examples:
      for candidate_node in train:
        if candidate_node in k_random_examples:
          continue
        if metric == 'euclidean':
          distance = euclidean(class_node.get_data(), candidate_node.get_data())
          if distance < candidate_node.get_min_distance():
            candidate_node.set_class(class_node.get_class())
            candidate_node.set_min_distance(distance)

def generate_dataset_from_labeled_nodes(train):
    classes = [node.get_class() for node in train]
    data = np.array([node.get_data() for node in train], dtype=np.float64)
    df = pd.DataFrame({
        'label': classes,
        'data': list(data)
    })
    return df  
    
# def get_means_from_labeled_data(data):
#     grouped = data.groupby('label')
#     next_centroids = []
#     for label, values in grouped:
#       new_mean_data = calculate_mean(values['data']) # Calculating the mean array for the grouped label (example: gets the new mean array for class-0, class-1...)
#       new_node = Node(new_mean_data)
#       new_node.set_class(label)
#       next_centroids.append(new_node)
#     return next_centroids

def calculate_mean(arrays):
  return np.mean(arrays, axis=0)

def calculate_downsample(dataset):
    for i, row in enumerate(dataset):
      downsampled_row = [] 
      left = 0
      right = 2
      
      while right <= len(row[1]):
        mean_value = sum(row[1][left:right]) / 2
        downsampled_row.append(mean_value)
        left += 2
        right += 2
      dataset[i][1] = downsampled_row
    return dataset

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

def show(filename, mode):
    dataset = read_data(filename)
    for obs in range(len(dataset)):
        for idx in range(784):
            if mode == 'pixels':
                if dataset[obs][1][idx] == '0':
                    print(' ', end='')
                else:
                    print('*', end='')
            else:
                print('%4s ' % dataset[obs][1][idx], end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % dataset[obs][0], end='')
        print(' ')

def print_conf_matrix(matrix):
  for row in matrix:
    print("     ", row)

def run_knn(train, test, valid, title="[ USING ORIGINAL DATASET WITHOUT DIMENSIONALITY REDUCTION ]"):
  """
  Parameters
  ----------
  train: The training data that KNN is learning from
  test: The test data that we run knn on
  valid: The validation data that we run knn on

  Returns
  -------
  The accuracy scores for the test and validation sets
  by the metric used (cosine similarity and euclidean distance)
  
  """
  print()
  print(title)
  print()
  print('     ----------------------------------')
  print("     K-Nearest-Neighbors ")
  print('     ----------------------------------')


  # VALIDATION SET
  print("     [  VALIDATION SET ]")
  valid_actual_cos, valid_pred_cos = knn(train, valid, 'cosim')
  valid_actual_euc, valid_pred_euc = knn(train, valid, 'euclidean')
  valid_actual_ham, valid_pred_ham = knn(train, valid, 'hamming')
  valid_actual_pc, valid_pred_pc = knn(train, valid, 'pearson_correlation')
  print("     Accuracy of Cosine Similarity KNN -- ", accuracy(valid_actual_cos, valid_pred_cos))
  print("     Confusion Matrix of Cosine Similarity KNN")
  print_conf_matrix(conf_matrix(valid_actual_cos, valid_pred_cos))
  print()
  print("     Accuracy of Euclidean KNN -- ", accuracy(valid_actual_euc, valid_pred_euc))
  print("     Confusion Matrix of Euclidean KNN")
  print_conf_matrix(conf_matrix(valid_actual_euc, valid_pred_euc))
  print()
  print("     Accuracy of Hamming KNN -- ", accuracy(valid_actual_ham, valid_pred_ham))
  print("     Confusion Matrix of Hamming KNN")
  print_conf_matrix(conf_matrix(valid_actual_ham, valid_pred_ham))
  print()
  print("     Accuracy of Pearson Correlation KNN -- ", accuracy(valid_actual_pc, valid_pred_pc))
  print("     Confusion Matrix of Pearson Correlation KNN")
  print_conf_matrix(conf_matrix(valid_actual_pc, valid_pred_pc))
  print()
  print()

  # TEST SET
  print("     [  TEST SET  ]")

  test_actual_cos, test_pred_cos = knn(train, test, 'cosim')
  test_actual_euc, test_pred_euc = knn(train, test, 'euclidean')
  print("     Accuracy of Cosine Similarity KNN -- ", accuracy(test_actual_cos, test_pred_cos))
  print("     Confusion Matrix of Cosine Similarity KNN")
  print_conf_matrix(conf_matrix(test_actual_cos, test_pred_cos))
  print("     Accuracy of Euclidean KNN")
  print("     ", accuracy(test_actual_euc, test_pred_euc))
  print("     Confusion Matrix of Euclidean KNN")
  print_conf_matrix(conf_matrix(test_actual_euc, test_pred_euc))
  
  print('\n\n\n')
  cosine_validation_accuracy = accuracy(valid_actual_cos, valid_pred_cos)
  euclidean_validation_accuracy = accuracy(valid_actual_euc, valid_pred_euc)
  hamming_validation_accuracy = accuracy(valid_pred_ham, valid_pred_ham)

  cosine_test_accuracy = accuracy(test_actual_cos, test_pred_cos)
  euclidean_test_accuracy = accuracy(test_actual_euc, test_pred_euc)

  return cosine_validation_accuracy, euclidean_validation_accuracy, cosine_test_accuracy, euclidean_test_accuracy, hamming_validation_accuracy

def run_kmeans(train, true_labels):
    print('     ----------------------------------')
    print("     K-Means")
    print('     ----------------------------------')
    cluster_memberships = kmeans(train, 'euclidean', k=10) # result with labels
    custom_score = metrics.silhouette_score(train, cluster_memberships)
    cami = metrics.adjusted_mutual_info_score(true_labels, cluster_memberships)
    cari = metrics.adjusted_rand_score(true_labels, cluster_memberships)
    print(f"Silhouette Score with k={10} (custom KMeans):", custom_score)
    print("Adjusted Mutual Information (AMI):", cami)
    print("Adjusted Rand Index (ARI):", cari)
    print('     ----------------------------------')
    print('\n\n\n')

def main():
    # show(filename, 'pixels')
    train_data = read_data('mnist_train.csv')
    valid_data = read_data('mnist_valid.csv')
    test_data = read_data('mnist_test.csv')

    inplace_min_max_scaling(train_data)
    inplace_min_max_scaling(test_data)
    inplace_min_max_scaling(valid_data)

    """
    ------------------------------------------------
    KNN with no dimensionality reduction
    ------------------------------------------------
    """
    no_dim_cosine_validation_accuracy, no_dim_euclidean_validation_accuracy, no_dim_cosine_test_accuracy, no_dim_euclidean_test_accuracy, hamming_validation_accuracy = run_knn(train_data, test_data, valid_data)
    

    """
    ------------------------------------------------
    KMeans with no dimensionality reduction
    ------------------------------------------------
    """
    kmeans_train_data = [row[1] for row in train_data] # Passing in flattened matrix without labels
    kmeans_train_labels = [row[0] for row in train_data] # Passing in flattened matrix without labels
    run_kmeans(kmeans_train_data, kmeans_train_labels)

    """
    ------------------------------------------------
    DOWNSAMPLING
    ------------------------------------------------
    """
    train_copy = train_data.copy()
    test_copy = test_data.copy()
    valid_copy = valid_data.copy()
    calculate_downsample(train_copy) # inplace calculation of downsample -- reduces the dataset by half exactly
    calculate_downsample(test_copy)
    calculate_downsample(valid_copy)

    downsampling_cosine_validation_accuracy, downsampling_euclidean_validation_accuracy, downsampling_cosine_test_accuracy, downsampling_euclidean_test_accuracy, downsampling_hamming_validation_accuracy = run_knn(train_copy, test_copy, valid_copy, title="[ DOWNSAMPLING - DIMENSIONALITY REDUCTION ]")


    """
    ------------------------------------------------
    Printing findings
    ------------------------------------------------
    """
    print("KNN with No Dimensionality Reduction:")
    print("-------------------------------------")
    print(f"Cosine Validation Accuracy: {no_dim_cosine_validation_accuracy}")
    print(f"Euclidean Validation Accuracy: {no_dim_euclidean_validation_accuracy}")
    print(f"Cosine Test Accuracy: {no_dim_cosine_test_accuracy}")
    print(f"Euclidean Test Accuracy: {no_dim_euclidean_test_accuracy}\n")
    print(f"Hamming Validation Accuracy: {hamming_validation_accuracy}\n")

    # KNN with Downsampling
    print("KNN with Downsampling Dimensionality Reduction:")
    print("-------------------------------------")
    print(f"Cosine Validation Accuracy: {downsampling_cosine_validation_accuracy}")
    print(f"Euclidean Validation Accuracy: {downsampling_euclidean_validation_accuracy}")
    print(f"Cosine Test Accuracy: {downsampling_cosine_test_accuracy}")
    print(f"Euclidean Test Accuracy: {downsampling_euclidean_test_accuracy}")

if __name__ == "__main__":
    main()
    
    # train_data = read_data('mnist_train.csv')
    # show('mnist_train.csv', 'pixels')
