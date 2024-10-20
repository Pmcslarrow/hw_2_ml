import numpy as np 
from collections import Counter
import heapq
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

class Node:
    def __init__(self, data):
        self.data = data
        self.min_distance = float('inf')
        self.class_label = None

    def get_data(self):
        return self.data
    
    def get_min_distance(self):
        return self.min_distance
    
    def get_class(self):
        return self.class_label

    def set_class(self, string):
        """
        Parameters
        ----------
        string: The string of the new class you want to assign
        """
        self.class_label = string

    def set_min_distance(self, new_min):
        """
        Parameters
        ----------
        new_min: The new lowest distance to this point (float)
        """
        self.min_distance = new_min


def accuracy(actuals, predicted):
    correct = 0
    for i in range(len(actuals)):
        if actuals[i] == predicted[i]:
            correct += 1
    return correct / len(actuals)

def conf_matrix(actuals, predicted):
    print('   0  1  2  3  4  5  6  7  8  9')
    return metrics.confusion_matrix(actuals, predicted)

def inplace_min_max_scaling(data):
    maximum = 255.0

    for i, row in enumerate(data):
        row = np.array(row[1], dtype=np.float64)
        scaled = row / maximum
        data[i][1] = scaled

def euclidean(a, b):
    """
    Returns Euclidean distance between vectors a and b
    """
    return np.sqrt(np.sum((np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64))**2))

def cosim(a, b):
    """
    Returns Cosine Similarity between vectors a and b
    """
    a = np.array(a, dtype=np.float64)  
    b = np.array(b, dtype=np.float64)  
    dot_product = np.dot(a, b)         
    return dot_product / (np.linalg.norm(a) * np.linalg.norm(b))

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

def knn(train, query, metric):
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

    k = 4
    print(f"Running KNN for k = {k} with {metric} metric")
    actuals = []
    predictions = []
    for test_example in query:
        k_distances = get_k_sorted_distances(test_example, train, metric=metric, k=k)
        most_common_label = get_most_common_label(k_distances)
        actuals.append(test_example[0])
        predictions.append(most_common_label)
    return actuals, predictions

def kmeans(train, query, metric):
    num_clusters = 9
    has_converged = False
    last_iteration_centroids = None
    next_iteration_centroids = None
    
    while not has_converged:
      nodes = initialize_nodes(train)  # Converts each row to a node representation
      if not next_iteration_centroids:
        centroids = select_random_centroids(nodes, num_clusters) # Gets the k random mean clusters from the training set
        last_iteration_centroids = centroids
      else:
        centroids = next_iteration_centroids
        last_iteration_centroids = centroids
      assign_labels_to_nodes(nodes, centroids, metric) # Assigns a class name to the node of the closest mean cluster
      updated_data = generate_dataset_from_labeled_nodes(nodes) # Creates a new dataset of label data and flattened representations    {class-8 ,   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...}
      next_iteration_centroids = get_means_from_labeled_data(updated_data)
      
      if last_iteration_centroids is not None and calculate_converge(last_iteration_centroids, next_iteration_centroids):
        has_converged = True
    return updated_data

def calculate_converge(last_centroids, next_centroids):
  epsilon = 1e-4
  for i in range(len(last_centroids)):
    difference = np.linalg.norm(np.array(last_centroids[i].get_data()) - np.array(next_centroids[i].get_data()))
    print(last_centroids[i].get_class(), difference)
    if difference >= epsilon:
      return False
  return True

def initialize_nodes(train):
    new_dataset = []
    for row in train:
        new_dataset.append(Node(row))
    return np.array(new_dataset)

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
    
def get_means_from_labeled_data(data):
    grouped = data.groupby('label')
    next_centroids = []
    for label, values in grouped:
      new_mean_data = calculate_mean(values['data']) # Calculating the mean array for the grouped label (example: gets the new mean array for class-0, class-1...)
      new_node = Node(new_mean_data)
      new_node.set_class(label)
      next_centroids.append(new_node)
    return next_centroids

def calculate_mean(arrays):
  return np.mean(arrays, axis=0)


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

def main():
    # show(filename, 'pixels')
    train_data = read_data('mnist_train.csv')
    valid_data = read_data('mnist_valid.csv')
    test_data = read_data('mnist_test.csv')

    inplace_min_max_scaling(train_data)
    inplace_min_max_scaling(test_data)

    """
    ------------------------------------------------
    K NEAREST NEIGHBORS
    ------------------------------------------------
    
    print("K-Nearest-Neighbors ")
    print('\n\n\n')

    # VALIDATION SET
    print('----------------------------------')
    valid_actual_cos, valid_pred_cos = knn(train_data, valid_data, 'cosim')
    valid_actual_euc, valid_pred_euc = knn(train_data, valid_data, 'euclidean')

    print("[  VALIDATION SET ]")
    print("Accuracy of Cosine Similarity KNN")
    print(accuracy(valid_actual_cos, valid_pred_cos))
    print("Confusion Matrix of Cosine Similarity KNN")
    print(conf_matrix(valid_actual_cos, valid_pred_cos))
    print("Accuracy of Euclidean KNN")
    print(accuracy(valid_actual_euc, valid_pred_euc))
    print("Confusion Matrix of Euclidean KNN")
    print(conf_matrix(valid_actual_euc, valid_pred_euc))
    print('----------------------------------')
    print()
    print()

    # TEST SET
    print('----------------------------------')
    test_actual_cos, test_pred_cos = knn(train_data, test_data, 'cosim')
    test_actual_euc, test_pred_euc = knn(train_data, test_data, 'euclidean')

    print("[  TEST SET  ]")
    print("Accuracy of Cosine Similarity KNN")
    print(accuracy(test_actual_cos, test_pred_cos))
    print("Confusion Matrix of Cosine Similarity KNN")
    print(conf_matrix(test_actual_cos, test_pred_cos))
    print("Accuracy of Euclidean KNN")
    print(accuracy(test_actual_euc, test_pred_euc))
    print("Confusion Matrix of Euclidean KNN")
    print(conf_matrix(test_actual_euc, test_pred_euc))
    print('----------------------------------')
    print('\n\n\n')
    """

    """
    ------------------------------------------------
    K MEANS
    ------------------------------------------------
    """
    means_train_data = [row[1] for row in train_data] # Passing in flattened matrix
    means_test_data = [row[1] for row in test_data]
    kmeans_resulting_dataset_with_clusters = kmeans(means_train_data, means_test_data, 'euclidean')
    print(kmeans_resulting_dataset_with_clusters)


if __name__ == "__main__":
    main()
