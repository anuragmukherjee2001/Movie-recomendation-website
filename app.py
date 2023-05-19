import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests


def fetch_poster(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=6faf5353af28d059faa9faeb938867d7&language=en-US".format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/original/" + data['poster_path']

def fetch_desc(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=6faf5353af28d059faa9faeb938867d7&language=en-US".format(movie_id))
    data = response.json()
    return data['overview']

def release_date(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=6faf5353af28d059faa9faeb938867d7&language=en-US".format(movie_id))
    data = response.json()
    return data['release_date']

def fetch_homepage(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=6faf5353af28d059faa9faeb938867d7&language=en-US".format(movie_id))
    data = response.json()
    return data['homepage']

def genre(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=6faf5353af28d059faa9faeb938867d7&language=en-US".format(movie_id))
    data = response.json()
    gen = data['genres']
    res = []
    for i in range(0,len(gen)):
        res.append(gen[i]['name'])

    return res    

df2 = pd.read_pickle(open('movies.pkl', 'rb'))
movies = df2['title'].values


indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
tfidf = TfidfVectorizer(stop_words='english')
df2['soup'] = df2['soup'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['soup'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)




def improved_recommendations(title):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

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
        return (v/(v+m) * R) + (m/(m+v) * C)

    qualified = movies[(movies['vote_count'] >= m) & (
        movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

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
    desc = []
    genres = []
    released_date = []
    homepage = []
    for i in df['title']:
        if(cnt != 5):
            col = st.columns(1)
            pred_movie.append(i)
            st.write(i)
            cnt += 1

    for i in df['id']:
        image_path.append(fetch_poster(i))
        desc.append(fetch_desc(i))
        genres.append(genre(i))
        released_date.append(release_date(i))
        homepage.append(fetch_homepage(i))
       

    for i in range(0,5):
        st.subheader(pred_movie[i])
        col1,col2 = st.columns(2)  
        
        with col1:
            
            st.image(image_path[i])
        with col2:
            st.write(desc[i])
            st.subheader("Genres")
            for i in range(0, len(genres[i])):
                st.write(genres[i][0])
            st.subheader("Release Date")    
            st.write(released_date[i]) 
        if homepage[i] is not '':
            st.subheader("Get the Movie")
            st.markdown(f'''
<a href={homepage[i]}><button style="background-color:GreenYellow;">Click Here</button></a>
''',
unsafe_allow_html=True)
        st.divider()      
