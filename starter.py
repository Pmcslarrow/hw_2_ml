import numpy as np 
from collections import Counter
import heapq
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    print('        0  1  2  3  4  5  6  7  8  9')
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
    print(f"     Running KNN for k = {k} with {metric} metric")
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
    #print(last_centroids[i].get_class(), difference)
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

def calculate_pca(dataset):
  labels = [item[0] for item in dataset]  
  features = np.array([item[1] for item in dataset])  
  pca = PCA(n_components=200)
  features_pca = pca.fit_transform(features)
  return [[str(label), np.array(features_pca[i], dtype=np.float64)] for i, label in enumerate(labels)]

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
    K NEAREST NEIGHBORS
    ------------------------------------------------
    """
    print()
    print("[ USING ORIGINAL DATASET WITHOUT DIMENSIONALITY REDUCTION ]")
    print()
    print('     ----------------------------------')
    print("     K-Nearest-Neighbors ")
    print('     ----------------------------------')


    # VALIDATION SET
    print("     [  VALIDATION SET ]")
    valid_actual_cos, valid_pred_cos = knn(train_data, valid_data, 'cosim')
    valid_actual_euc, valid_pred_euc = knn(train_data, valid_data, 'euclidean')
    print("     Accuracy of Cosine Similarity KNN -- ", accuracy(valid_actual_cos, valid_pred_cos))
    print("     Confusion Matrix of Cosine Similarity KNN")
    print_conf_matrix(conf_matrix(valid_actual_cos, valid_pred_cos))
    print()
    print("     Accuracy of Euclidean KNN -- ", accuracy(valid_actual_euc, valid_pred_euc))
    print("     Confusion Matrix of Euclidean KNN")
    print_conf_matrix(conf_matrix(valid_actual_euc, valid_pred_euc))
    print()
    print()

    # TEST SET
    print("     [  TEST SET  ]")

    test_actual_cos, test_pred_cos = knn(train_data, test_data, 'cosim')
    test_actual_euc, test_pred_euc = knn(train_data, test_data, 'euclidean')
    print("     Accuracy of Cosine Similarity KNN -- ", accuracy(test_actual_cos, test_pred_cos))
    print("     Confusion Matrix of Cosine Similarity KNN")
    print_conf_matrix(conf_matrix(test_actual_cos, test_pred_cos))
    print("     Accuracy of Euclidean KNN")
    print("     ", accuracy(test_actual_euc, test_pred_euc))
    print("     Confusion Matrix of Euclidean KNN")
    print_conf_matrix(conf_matrix(test_actual_euc, test_pred_euc))
    
    print('\n\n\n')
    

    """
    ------------------------------------------------
    K MEANS
    ------------------------------------------------
    """
    print('     ----------------------------------')
    print("     K-Means")
    print('     ----------------------------------')
    means_train_data = [row[1] for row in train_data] # Passing in flattened matrix without labels
    means_test_data = [row[1] for row in test_data]
    kmeans_resulting_dataset_with_clusters = kmeans(means_train_data, means_test_data, 'euclidean') # result with labels
    print("     ", kmeans_resulting_dataset_with_clusters) 
    print('     ----------------------------------')
    print('\n\n\n')




    """
    ------------------------------------------------
    PRINCIPAL COMPONENT ANALYSIS
    ------------------------------------------------
    """
    train_copy = train_data.copy()
    test_copy = test_data.copy()
    valid_copy = valid_data.copy()

    df_train_pca = calculate_pca(train_copy)
    df_test_pca = calculate_pca(test_copy)
    df_valid_pca = calculate_pca(valid_copy)

    print("[ PCA - DIMENSIONALITY REDUCTION  ]")
    print()
    print('     ----------------------------------')
    print("     K-Nearest-Neighbors ")
    print('     ----------------------------------')

        # VALIDATION SET
    print("     [  VALIDATION SET ]")
    pca_valid_actual_cos, pca_valid_pred_cos = knn(df_train_pca, df_valid_pca, 'cosim')
    pca_valid_actual_euc, pca_valid_pred_euc = knn(df_train_pca, df_valid_pca, 'euclidean')
    print("     Accuracy of Cosine Similarity KNN -- ", accuracy(pca_valid_actual_cos, pca_valid_pred_cos))
    print("     Confusion Matrix of Cosine Similarity KNN")
    print_conf_matrix(conf_matrix(pca_valid_actual_cos, pca_valid_pred_cos))
    print()
    print("     Accuracy of Euclidean KNN -- ", accuracy(pca_valid_actual_euc, pca_valid_pred_euc))
    print("     Confusion Matrix of Euclidean KNN")
    print_conf_matrix(conf_matrix(pca_valid_actual_euc, pca_valid_pred_euc))
    print()
    print()

    # TEST SET
    print("     [  TEST SET  ]")
    pca_test_actual_cos, pca_test_pred_cos = knn(df_train_pca, df_test_pca, 'cosim')
    pca_test_actual_euc, pca_test_pred_euc = knn(df_train_pca, df_test_pca, 'euclidean')
    print("     Accuracy of Cosine Similarity KNN -- ", accuracy(pca_test_actual_cos, pca_test_pred_cos))
    print("     Confusion Matrix of Cosine Similarity KNN")
    print_conf_matrix(conf_matrix(pca_test_actual_cos, pca_test_pred_cos))
    print("     Accuracy of Euclidean KNN")
    print("     ", accuracy(pca_test_actual_euc, pca_test_pred_euc))
    print("     Confusion Matrix of Euclidean KNN")
    print_conf_matrix(conf_matrix(pca_test_actual_euc, pca_test_pred_euc))
    print('\n\n\n')



    """
    ------------------------------------------------
    DOWNSAMPLING
    ------------------------------------------------
    """
    print("[ DOWNSAMPLING - DIMENSIONALITY REDUCTION ]")
    calculate_downsample(train_copy) # inplace calculation of downsample -- reduces the dataset by half exactly
    calculate_downsample(test_copy)
    calculate_downsample(valid_copy)
    print()
    print('     ----------------------------------')
    print("     K-Nearest-Neighbors ")
    print('     ----------------------------------')

        # VALIDATION SET
    print("     [  VALIDATION SET ]")
    downsample_valid_actual_cos, downsample_valid_pred_cos = knn(train_copy, valid_copy, 'cosim')
    downsample_valid_actual_euc, downsample_valid_pred_euc = knn(train_copy, valid_copy, 'euclidean')
    print("     Accuracy of Cosine Similarity KNN -- ", accuracy(downsample_valid_actual_cos, downsample_valid_pred_cos))
    print("     Confusion Matrix of Cosine Similarity KNN")
    print_conf_matrix(conf_matrix(downsample_valid_actual_cos, downsample_valid_pred_cos))
    print()
    print("     Accuracy of Euclidean KNN -- ", accuracy(downsample_valid_actual_euc, downsample_valid_pred_euc))
    print("     Confusion Matrix of Euclidean KNN")
    print_conf_matrix(conf_matrix(downsample_valid_actual_euc, downsample_valid_pred_euc))
    print()
    print()

    # TEST SET
    print("     [  TEST SET  ]")
    downsample_test_actual_cos, downsample_test_pred_cos = knn(train_copy, test_copy, 'cosim')
    downsample_test_actual_euc, downsample_test_pred_euc = knn(train_copy, test_copy, 'euclidean')
    print("     Accuracy of Cosine Similarity KNN -- ", accuracy(downsample_test_actual_cos, downsample_test_pred_cos))
    print("     Confusion Matrix of Cosine Similarity KNN")
    print_conf_matrix(conf_matrix(downsample_test_actual_cos, downsample_test_pred_cos))
    print("     Accuracy of Euclidean KNN")
    print("     ", accuracy(downsample_test_actual_euc, downsample_test_pred_euc))
    print("     Confusion Matrix of Euclidean KNN")
    print_conf_matrix(conf_matrix(downsample_test_actual_euc, downsample_test_pred_euc))
    print('\n\n\n')


if __name__ == "__main__":
    main()
