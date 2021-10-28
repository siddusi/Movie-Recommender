#import pandas 
import pandas as pd

#load movies metadata into pandas dataframe
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

#print the first 3 rows
print('The first 3 rows of the dataframe are:')
print(str(metadata.head(3)))

#now, let's apply the weighted rating formula to see which movies are eligible based on sufficient vote numbers
C = metadata['vote_average'].mean()                                             #calculate the mean of vote average column 
print("\nAverage rating on IMDB: " + str(C))                                    #avg rating on IMDB is around a 5.6

#now let's calculate the number of votes (m) recieved by a movie in the 90th percentile
m = metadata['vote_count'].quantile(0.90)
print('The minimum number of votes required to be in the chart: ' + str(m) +' votes')

#make a copy of the original metadata data frame to ensure your changes don't affect original data frame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]                     #filter out all qualified movies into new DataFrame 
print('The tuple that represents the dimensionality of the DataFrame - q_movies is: ' + str(q_movies.shape))

#now, we calculate the weighted rating for each QUALIFIED movie 
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']                     #v is the number of votes for the movie
    R = x['vote_average']                   #R is the average rating of the movie
    return (v/(v+m) * R) + (m/(m+v) * C)    #the weighted rating is calculated here

#score is a new feature where we apply the weighted_rating function on all of our qualified movies
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#sort the movies in descending score (highest to lowest score)
q_movies = q_movies.sort_values('score', ascending=False)

#print the top 20 movies and their data
print('The top 20 movies are: ')
print(str(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20)))