import streamlit as st
import pickle5 as pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

def fetch_poster(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=6faf5353af28d059faa9faeb938867d7&language=en-US".format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/original/" + data['poster_path']

df2 = pickle.load(open('movies.pkl', 'rb'))
movies = df2['title'].values


indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
tfidf = TfidfVectorizer(stop_words='english')
df2['soup'] = df2['soup'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['soup'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)




def improved_recommendations(title):
    # Getting The index of the Input Flim
    idx = indices[title]

    # Getting the Cosine Similarity Tuple of the selected input flim
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sorting the Similarity Scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Taking The Top 25 Similairty Scores & movie indices
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    # Getting The Title, Vote Count, Vote Average
    movies = df2.iloc[movie_indices][['id','title', 'vote_count', 'vote_average']]
    vote_counts = movies[movies['vote_count'].notnull()
                         ]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull(
    )]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)

    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)

    # Filtering the Movies Vote Count > 0,60 Quantile, Greater than average rating
    qualified = movies[(movies['vote_count'] >= m) & (
        movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    # Getting the IMDB Weighted Score
    qualified['weighted_score'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values(
        'weighted_score', ascending=False).head(10)
    return qualified


st.title("Movie Recommender System")

option = st.selectbox("Select Movie Name", movies)

if st.button('Recommend'):
    df = improved_recommendations(option)
    # st.write(df['id'])


    cnt = 0
    pred_movie = []
    image_path = []
    for i in df['title']:
        if(cnt != 5):
            col = st.columns(1)
            pred_movie.append(i)
            st.write(i)
            cnt += 1

    for i in df['id']:
        image_path.append(fetch_poster(i))
    

    col1,col2,col3,col4, col5 = st.columns(5)
    with col1:
        st.text(pred_movie[0])
        st.image(image_path[0])
    with col2:
        st.text(pred_movie[1])
        st.image(image_path[1])
    with col3:
        st.text(pred_movie[2])
        st.image(image_path[2])      
    with col4:
        st.text(pred_movie[3])
        st.image(image_path[3]) 
    with col5:
        st.text(pred_movie[4])
        st.image(image_path[4])    
