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

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from starter import read_file, calculate_similarities, impute_rating

if __name__ == '__main__':
    training_datasets = read_file(prefix="train")
    test_datasets = read_file(prefix='test')
    validation_datasets = read_file(prefix='valid')

    # Joining all the data together
    training_data = pd.concat(training_datasets, ignore_index=True)
    test_data = pd.concat(test_datasets, ignore_index=True)
    valid_data = pd.concat(validation_datasets, ignore_index=True)

    # Encoding the genre, gender, and occupation for each dataset
    genre_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()
    occupation_encoder = LabelEncoder()

    training_data['genre'] = genre_encoder.fit_transform(training_data['genre'])
    training_data['gender'] = gender_encoder.fit_transform(training_data['gender'])
    training_data['occupation'] = occupation_encoder.fit_transform(training_data['occupation'])

    test_data['genre'] = genre_encoder.transform(test_data['genre'])
    test_data['gender'] = gender_encoder.transform(test_data['gender'])
    test_data['occupation'] = occupation_encoder.transform(test_data['occupation'])

    valid_data['genre'] = genre_encoder.transform(valid_data['genre'])
    valid_data['gender'] = gender_encoder.transform(valid_data['gender'])
    valid_data['occupation'] = occupation_encoder.transform(valid_data['occupation'])

    # Creating User-based collaborative filtering with a pivot table
    train_pivot = training_data.pivot(index='user_id', columns='movie_id', values='rating')
    training_user_similarities = calculate_similarities(train_pivot)
    
    train_demo_df = training_data.loc[:, ['user_id', 'age', 'gender', 'occupation']].drop_duplicates().set_index('user_id')
    training_user_demo_similarities = calculate_similarities(train_demo_df)

    recommendations = impute_rating(train_pivot, training_user_similarities, similarity_dict_demo=training_user_demo_similarities)
    # recommendations = impute_rating(train_pivot, training_user_similarities, similarity_dict_demo=None)

    import pickle

    with open('./recommendations.pkl', 'wb') as f:
      pickle.dump(recommendations, f)

    training_data.to_csv('./training_data.csv', index=False)
    test_data.to_csv('./test_data.csv', index=False)
    valid_data.to_csv('./valid_data.csv', index=False)

    for user_id, recommendation_list in recommendations.items():
      print(f"User ID {user_id} SHOULD WATCH:")
      for recommendation in recommendation_list:
        movie_id = recommendation[1]
        movie_details = training_data[training_data['movie_id'] == movie_id].values[0]
        movie_title = movie_details[3] 
        print(f'--- {movie_title}')
      print()