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

|
| PIVOT THE DATA INTO THIS (BUT WITH ALL USERS) Movie_id on top, user_id's are the rows, and the count are the ratings
v

movie_id  56  171  580  592  606  904  1470  1478  1582  184
user_id                                                     
405       3   0    0    0    2    0    1    0    0    0

"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import collections

def euclidean(a, b):
  """
  Parameters
  ----------
  The pivoted row of ratings from user1 (a) and user2 (b)

  Returns
  -------
  - The euclidean distance between the two users
  - The movie IDs that user 1 rated but user 2 did not (can use this info for recommendations)
  - The movie IDs that user 2 rated but user 1 did not (can use this info for recommendations)  
  """

  # Drop the NA values from a and b and find where both users have rated the SAME movie
  movie_ids_no_nan_a = a.dropna().index
  movie_ids_no_nan_b = b.dropna().index
  movie_ids_both_rated = (movie_ids_no_nan_a.intersection(movie_ids_no_nan_b)).tolist()

  # Movies that each has ranked that the other has not ranked (these can be used as our recommendations)
  movie_ids_a_ranked_not_b = (movie_ids_no_nan_a.difference(movie_ids_no_nan_b)).tolist()
  movie_ids_b_ranked_not_a = (movie_ids_no_nan_b.difference(movie_ids_no_nan_a)).tolist()

  ratings_a = a[movie_ids_both_rated]
  ratings_b = b[movie_ids_both_rated]
  
  distance = np.sqrt(np.sum((ratings_a - ratings_b)**2))
  return distance, movie_ids_a_ranked_not_b, movie_ids_b_ranked_not_a

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

def calculate_similarities(data):
  """
  Parameters
  ----------
  data: Pivoted dataframe where movie_ids are the columns and the rows are each user and the values are the ratings per movie

  Returns
  -------
  Returns a dictionary that represents the similarity between each user... It looks spooky, but it is as simple as this:
  { user_1: [(distance, user_2, movie ids user 1 rated but user 2 did not, movie ids user 2 rated but user 1 did not)]}
  """
  similarities = collections.defaultdict(list)
  visited = set()
  users = data.index.tolist() 
  for i in range(len(users)):
    for j in range(len(users)):
      user_1 = users[i]
      user_2 = users[j]
      if user_1 != user_2 and tuple(sorted((user_1, user_2))) not in visited:
        visited.add(tuple(sorted((user_1, user_2))))
        distance, movie_ids_user1_not_user2, movie_ids_user2_not_user1 = euclidean(data.loc[user_1], data.loc[user_2])
        similarities[user_1].append((distance, user_2, movie_ids_user1_not_user2, movie_ids_user2_not_user1))
        similarities[user_2].append((distance, user_1, movie_ids_user1_not_user2, movie_ids_user2_not_user1))
  return similarities


if __name__ == '__main__':
    training_datasets = read_file(prefix="train")
    test_datasets = read_file(prefix='test')
    validation_datasets = read_file(prefix='valid')

    # Joining all the data together
    training_data = pd.concat(training_datasets, ignore_index=True)
    test_data = pd.concat(test_datasets, ignore_index=True)
    valid_data = pd.concat(validation_datasets, ignore_index=True)

    # Creating User-based collaborative filtering
    train_pivot = training_data.pivot(index='user_id', columns='movie_id', values='rating')
    training_user_similarities = calculate_similarities(train_pivot)

    for user_id, distance_tuple in training_user_similarities.items():
      print(f"User ID: {user_id}")
      for distance, other_user, _, _ in distance_tuple:
        print(f"- Distance to user ID {other_user} == {distance}")
      print()