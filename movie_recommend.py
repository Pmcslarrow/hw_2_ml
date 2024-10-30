"""
NOTE:

Step 1: 
  Calculate the similarity between each users. 
  You can use either distance measurement, but make sure 
  you have a representation of each user and a list of users most similar
  Also, you could try centering the data around 0 for scaling.

Step 2:
  Next you calculate the Rating, R, that a user, U, 
  would give to a certain item I (that they haven't seen). 
  This means you could learn how likely you are to watch a movie
  from the average rating given by n users who have seen it.
  For example, you could pick the 10 closest users to you,
  and see what they rated the movie. Add them all up. Then divide by 10. 
  Gets you the average score that you may want to see this movie. 
  You can either do this average, or you could do a weighted average
  where the first closest user has more of a factor in the score than
  the second cloest, etc...
  
Example data:
user_id 	movie_id	  rating	  title	              genre	  age	  gender	  occupation
405	      43	        1	        Disclosure (1994)	  Drama	  22	  F	        healthcare


     user_id  movie_id  rating  title  genre  age  gender  occupation
0        405        56       3    223      5    0       0           0
1        405       592       0    310     12    0       0           0
2        405      1582       0    285      9    0       0           0
3        405       171       0     86     14    0       0           0
4        405       580       0    102      4    0       0           0
..       ...       ...     ...    ...    ...  ...     ...         ...
332      405       904       0    179      4    0       0           0
333      405       606       2     10      7    0       0           0
334      405      1470       1    132      2    0       0           0
335      405      1478       0     83      7    0       0           0
336      405       184       0     22     10    0       0           0

[337 rows x 8 columns]
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def euclidean(a, b):
    """
    Returns Euclidean distance between vectors a and b
    """
    return np.sqrt(np.sum((np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64))**2))

def read_file(prefix="train"):
  """
  Parameters
  ----------
  prefix: The prefix of the filename you want --> train_a.txt  train is the prefix here and we assume a, b, and c are the only postfix

  Returns
  -------
  An array containing encoded pandas DataFrames for a, b, and c of your respective prefix filename
  For example, training_data[0] contains train_a.txt as a pandas DataFrame
  
  """
  training_datasets = [] # train_a.txt, train_b.txt, train_c.txt in pandas DataFrame format (encoded)
  postfix=['a', 'b', 'c']
  for letter in postfix:
    filename = f"{prefix}_{letter}.txt"
    data = pd.read_csv(filename, sep='\t')
    label_encoder = LabelEncoder()

    for column_name in data.columns:
      if column_name not in ['user_id', 'movie_id']:
        data[column_name] = label_encoder.fit_transform(data[column_name])
    training_datasets.append(data)
  return training_datasets

if __name__ == '__main__':
    training_data = read_file(prefix="train")
    test_data = read_file(prefix='test')
    validation_data = read_file(prefix='valid')

    a = training_data[0].iloc[0, 2:]
    b = training_data[0].iloc[1, 2:]

    print(euclidean(a, b))

    #print(training_data[0])