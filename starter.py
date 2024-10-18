import numpy as np 
from collections import Counter
import heapq

def accuracy(actuals, predicted):
  correct = 0
  for i in range(len(actuals)):
    if actuals[i] == predicted[i]:
      correct += 1
  return correct / len(actuals)

def inplace_min_max_scaling(data):
  maximum = 255.0

  for i, row in enumerate(data):
    row = np.array(row[1], dtype=np.float64)
    scaled = row / maximum
    data[i][1] = scaled
  


def euclidean(a,b):
  """
  Returns Euclidean distance between vectors and b
  """
  return np.sqrt(np.sum((np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64))**2))
        
def cosim(a, b):
  """
  Returns Cosine Similarity between vectors and b
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
  inplace_min_max_scaling(train)
  inplace_min_max_scaling(query)

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


# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    return(labels)

def read_data(file_name):
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    #show(filename,'pixels')
    train_data = read_data('mnist_train.csv')
    valid_data = read_data('mnist_valid.csv')
    test_data = read_data('mnist_test.csv')

    # VALIDATION SET
    print('----------------------------------')
    valid_actual_cos, valid_pred_cos = knn(train_data, valid_data, 'cosim')
    valid_actual_euc, valid_pred_euc = knn(train_data, valid_data, 'euclidean')

    print("[  VALIDATION SET ]")
    print("Accuracy of Cosine Similarity KNN")
    print(accuracy(valid_actual_cos, valid_pred_cos))
    print("Accuracy of Euclidean KNN")
    print(accuracy(valid_actual_euc, valid_pred_euc))
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
    print("Accuracy of Euclidean KNN")
    print(accuracy(test_actual_euc, test_pred_euc))
    print('----------------------------------')
    
if __name__ == "__main__":
    main()
    