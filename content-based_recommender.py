#import pandas 
import pandas as pd
#import scikit-learn
import sklearn
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# Import linear_kernel for the cosine similarity
from sklearn.metrics.pairwise import linear_kernel

#Purpose: to recommend movies that are similar to a particular movie

#load movies metadata into pandas dataframe
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

#this prints the plot overviews of the first 5 movies in the data frame
print('\nThe main plot overviews of the first 5 movies in the data frame are: ')
print(str(metadata['overview'].head()))

#Define a TF-IDF Vectorizer Object. 
#Remove all english stop words such as 'the', 'a' and downweight the common words to have less skewed data
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string (Replace not-a-number values with a blank string;)
metadata['overview'] = metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Output the shape of tfidf_matrix
print('The shape of the tfidf_matrix is: ' + str(tfidf_matrix.shape))

#Array mapping from feature integer indices to feature name.
#print(str(tfidf.get_feature_names()[5000:5010]))

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles as a mechanism to get index of movie given title
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

#now, build a fucntion that takes a movie title as input and outputs similar movies to it
def get_recommendations(title, cosine_sim = cosine_sim):
    idx = indices[title]                                                # Get the index of the movie that matches the title
    sim_scores = list(enumerate(cosine_sim[idx]))                       # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)   # sort the movies based on their similarity scores
    sim_scores = sim_scores[1:11]                                       # get the scores of the first 10 similar movies 
    movie_indices = [i[0] for i in sim_scores]                          # get the movie indices
    
    return metadata['title'].iloc[movie_indices]                        # return the top 10 most similar movies

#test function
print(str(get_recommendations('The Dark Knight Rises')))