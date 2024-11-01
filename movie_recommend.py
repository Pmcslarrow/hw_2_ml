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
13        n   n    n    n
655

"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import collections
import heapq

def cosim(a, b):
  # Drop the NA values from a and b and find where both users have rated the SAME movie
  movie_ids_no_nan_a = a.dropna().index
  movie_ids_no_nan_b = b.dropna().index
  movie_ids_both_rated = (movie_ids_no_nan_a.intersection(movie_ids_no_nan_b)).tolist()

  # Movies that each has ranked that the other has not ranked (these can be used as our recommendations)
  movie_ids_a_ranked_not_b = (movie_ids_no_nan_a.difference(movie_ids_no_nan_b)).tolist()
  movie_ids_b_ranked_not_a = (movie_ids_no_nan_b.difference(movie_ids_no_nan_a)).tolist()

  ratings_a = a[movie_ids_both_rated].to_numpy()
  ratings_b = b[movie_ids_both_rated].to_numpy()

  norm_a = np.linalg.norm(ratings_a)
  norm_b = np.linalg.norm(ratings_b)
  
  if norm_a == 0 or norm_b == 0:
    return 0.0

  dot_product = np.dot(ratings_a, ratings_b)
  print("DOT: ", dot_product)
  similarity = (dot_product / (norm_a * norm_b))
  return similarity

def calculate_similarities(data):
  """
  Parameters
  ----------
  data: Pivoted dataframe where movie_ids are the columns and the rows are each user and the values are the ratings per movie

  Returns
  -------
  NOTE COME BACK
  """
  similarities = collections.defaultdict(lambda: collections.defaultdict(float))
  visited = set()
  users = data.index.tolist() 
  for i in range(len(users)):
    for j in range(len(users)):
      user_1 = users[i]
      user_2 = users[j]
      if user_1 != user_2 and tuple(sorted((user_1, user_2))) not in visited:
        visited.add(tuple(sorted((user_1, user_2))))
        distance = cosim(data.loc[user_1], data.loc[user_2])
        similarities[user_1][user_2] = distance
        similarities[user_2][user_1] = distance
        # similarities[user_1].append((distance, user_2))
        # similarities[user_2].append((distance, user_1))
  return similarities

def impute_rating(pivot_table, similarity_dict):
  rows, cols = pivot_table.shape
  recommendations = collections.defaultdict(list)

  for movie_id in pivot_table.columns:
    ratings_per_movie_id = pivot_table[movie_id]
    nan_count = ratings_per_movie_id.isna().sum()
    difference = rows - nan_count
    if difference < 2:
      continue

    builder = {}
    for user_id, rating in ratings_per_movie_id.items():
      builder[user_id] = rating
    
    for user_id, rating in builder.items():
      if np.isnan(rating):
        continue

      imputed_rating_num = 0
      imputed_rating_denom = 0
      
      for other_user_id, other_rating in builder.items():
        if other_user_id == user_id or np.isnan(other_rating):
          continue
        imputed_rating_num += (other_rating * similarity_dict[user_id][other_user_id])
        imputed_rating_denom += similarity_dict[user_id][other_user_id]

      recommendation = imputed_rating_num / imputed_rating_denom
      recommendation = min(max(recommendation, 0), 5)
      heapq.heappush(recommendations[user_id], (recommendation, movie_id))
      
      if len(recommendations[user_id]) > 5:
        heapq.heappop(recommendations[user_id])
  return recommendations

      
    
      
      
  



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
    training_datasets.append(data)
  return training_datasets


if __name__ == '__main__':
    training_datasets = read_file(prefix="train")
    test_datasets = read_file(prefix='test')
    validation_datasets = read_file(prefix='valid')

    # Joining all the data together
    training_data = pd.concat(training_datasets, ignore_index=True)
    test_data = pd.concat(test_datasets, ignore_index=True)
    valid_data = pd.concat(validation_datasets, ignore_index=True)

    genre_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()
    occupation_encoder = LabelEncoder()

    training_data['genre'] = genre_encoder.fit_transform(training_data['genre'])
    training_data['gender'] = gender_encoder.fit_transform(training_data['gender'])
    training_data['occupation'] = occupation_encoder.fit_transform(training_data['occupation'])

    # Creating User-based collaborative filtering
    train_pivot = training_data.pivot(index='user_id', columns='movie_id', values='rating')
    training_user_similarities = calculate_similarities(train_pivot)
    

    impute_rating(train_pivot, training_user_similarities)

    # for user_id, distance_tuple in training_user_similarities.items():
    #   print(f"User ID: {user_id}")
    #   for distance, other_user, _, _ in distance_tuple:
    #     print(f"- Distance to user ID {other_user} == {distance}")
    #   print()