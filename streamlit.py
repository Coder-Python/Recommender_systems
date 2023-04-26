# Imports
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st

# Load preprocessed movie dataset
movies = pd.read_csv("./data/movies_metadata_preprocessed.csv")

# Get title of movie
def get_title(index):
    return movies[movies.index == index]["title"].values[0]

# Get index of movie
def get_index(title):
    return movies[movies.title == title]["index"].values[0]

# Compute the similarity matrix and store it in the cache for Streamlit
@st.cache_resource
def compute_similarity_matrix():
    # Load pre-trained model
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Get Embeddings for movie overviews
    sentence_embeddings = bert.encode(movies["overview"].tolist())

    # Compute similarity between movie overviews
    similarity = cosine_similarity(sentence_embeddings)

    return similarity

# Compute similarity matrix
similarity = compute_similarity_matrix()

# Streamlit app
# Define app title
st.title("Movie Recommendation App")

# Page appearance and background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1564115484-a4aaa88d5449");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

# Page configuration for HTML/CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Create an input field for the user to enter a movie with autocompletion feature
user_movie = st.selectbox("Enter the name of a movie :", movies["title"].tolist())

# Create a submit button to trigger the recommendation code
if st.button("Get Recommendations"):
    # Perform the recommendation and display the results
    recommendations = sorted(list(enumerate(similarity[get_index(user_movie)])), key=lambda x: x[1], reverse=True)
    st.write(f"The top 3 recommendations for {user_movie} are :")
    # Output the top 3 recommended movies
    for i in range(1, 4):
        recommended_movie_title = get_title(recommendations[i][0])
        recommended_movie_overview = movies.loc[movies["title"] == recommended_movie_title, "overview"].iloc[0]
        st.write(f"{recommended_movie_title} : {recommended_movie_overview}")
