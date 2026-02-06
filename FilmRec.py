import streamlit as st
import pandas as pd
from collections import defaultdict
from surprise import Reader, Dataset, SVD, KNNBasic

# Load datasets
interactions_df = pd.read_csv('C:\\Users\\malin\\Documents\\Veracity\\Film Rec\\interactions.csv')
item_features_df = pd.read_csv('C:\\Users\\malin\\Documents\\Veracity\\Film Rec\\item_features.csv')
user_data_df = pd.read_csv('C:\\Users\\malin\\Documents\\Veracity\\Film Rec\\user_data.csv')

# Convert user_id and movie_id to string to ensure compatibility with surprise
interactions_df['user_id'] = interactions_df['user_id'].astype(str)
interactions_df['movie_id'] = interactions_df['movie_id'].astype(str)
item_features_df['movie_id'] = item_features_df['movie_id'].astype(str)
user_data_df['user_id'] = user_data_df['user_id'].astype(str)
user_data_df['user_name'] = user_data_df['user_name'].str.lower()  # Convert usernames to lowercase

# Extract the first genre from the genre list
item_features_df['first_genre'] = item_features_df['genres'].str.split(',').str[0].str.strip()

# Merge interactions with item features to get movie titles
interactions_with_titles_df = pd.merge(interactions_df[['user_id', 'movie_id']], item_features_df[['movie_id', 'title', 'first_genre']], on='movie_id', how='left')

# Define a Reader and load the dataset into a Dataset object for surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interactions_df[['user_id', 'movie_id', 'points']], reader)

# Use KNNBasic for item-based filtering
sim_options = {
    'name': 'cosine',
    'user_based': False  # item-based filtering
}
model = KNNBasic(sim_options=sim_options)
trainset = data.build_full_trainset()
model.fit(trainset)

# Function to get top-N recommendations based on a movie
def get_similar_movies(movie_id, n=10):
    # Get the inner id of the movie
    inner_id = model.trainset.to_inner_iid(movie_id)
    # Get similar movies
    neighbors = model.get_neighbors(inner_id, k=n)
    # Convert inner ids to movie ids
    movie_neighbors = [model.trainset.to_raw_iid(inner_id) for inner_id in neighbors]
    return movie_neighbors

# Function to get top-N recommendations for a user
def get_top_n_recommendations(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Sort the predictions for each user and retrieve the highest rated items
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for iid, _ in user_ratings[:n]]  # Only keep the movie IDs

    return top_n

# Streamlit UI
st.title('Movie Recommendation System')

# Option to select the recommendation type
recommendation_type = st.radio('Select Recommendation Type', ('By User', 'By Movie'))

if recommendation_type == 'By User':
    # Input user ID or username
    user_input = st.text_input('Enter User ID or Username:', '').strip().lower()

    if user_input:
        # Check if input is a username
        user_row = user_data_df[user_data_df['user_name'] == user_input]
        
        if not user_row.empty:
            user_id = user_row['user_id'].values[0]
            user_name = user_row['user_name'].values[0].title()  # Capitalize for display
        else:
            # Check if input is a user ID
            user_row = user_data_df[user_data_df['user_id'].str.lower() == user_input]
            
            if not user_row.empty:
                user_id = user_row['user_id'].values[0]
                user_name = user_row['user_name'].values[0].title()  # Capitalize for display
            else:
                st.error('User not found.')
                user_id = None
                user_name = None

        if user_id:
            # Show movies the user has interacted with
            user_interactions = interactions_with_titles_df[interactions_with_titles_df['user_id'] == user_id]
            st.subheader(f'Movies {user_name} Has Interacted With')
            st.dataframe(user_interactions[['title', 'first_genre']])

            # Predict ratings for all movies that the user hasn't rated yet
            all_movies = set(interactions_df['movie_id'].unique())
            movies_rated_by_user = set(user_interactions['movie_id'].unique())
            movies_to_predict = list(all_movies - movies_rated_by_user)

            # Create a testset for the user
            testset = [[user_id, movie_id, 4.] for movie_id in movies_to_predict]  # 4. is a dummy rating value
            user_predictions = model.test(testset)

            # Get top-N recommendations for the user
            top_n_recommendations = get_top_n_recommendations(user_predictions, n=10)

            # Retrieve titles and genres for the recommended movies
            recommended_movie_ids = top_n_recommendations[user_id]
            recommended_movies = item_features_df[item_features_df['movie_id'].isin(recommended_movie_ids)]

            # Genre filtering
            genres = st.multiselect('Select Genres', item_features_df['first_genre'].unique())
            if genres:
                recommended_movies = recommended_movies[recommended_movies['first_genre'].isin(genres)]

            recommended_movies = recommended_movies.rename(columns={'title': 'Movie Title', 'first_genre': 'Genre'})
            # Display the top-N recommendations with genres
            st.subheader(f'Top 10 Recommendations for {user_name}')
            st.dataframe(recommended_movies[['Movie Title', 'Genre']])

elif recommendation_type == 'By Movie':
    # Input movie title for recommendations
    movie_input = st.text_input('Enter Movie Title:', '').strip()

    if movie_input:
        # Check if the movie title exists
        movie_row = item_features_df[item_features_df['title'].str.lower() == movie_input.lower()]
        
        if not movie_row.empty:
            movie_id = movie_row['movie_id'].values[0]
            similar_movie_ids = get_similar_movies(movie_id)
            similar_movies = item_features_df[item_features_df['movie_id'].isin(similar_movie_ids)]
            similar_movies = similar_movies.rename(columns={'title': 'Movie Title', 'first_genre': 'Genre'})
            st.subheader(f'Recommendations based on "{movie_input}"')
            st.dataframe(similar_movies[['Movie Title', 'Genre']])
        else:
            st.error('Movie not found.')

# Calculate trending movies based on interaction count
trending_movies_df = interactions_df.groupby('movie_id').size().reset_index(name='interaction_count')
trending_movies_df = trending_movies_df.sort_values(by='interaction_count', ascending=False)
trending_movies_df = trending_movies_df.head(10)  # Top 10 trending movies
trending_movies_df = pd.merge(trending_movies_df, item_features_df[['movie_id', 'title', 'first_genre']], on='movie_id', how='left')

# Display trending movies
st.subheader('Trending Movies')
trending_movies_df = trending_movies_df.rename(columns={'title': 'Movie Title', 'first_genre': 'Genre'})
st.dataframe(trending_movies_df[['Movie Title', 'Genre']])
