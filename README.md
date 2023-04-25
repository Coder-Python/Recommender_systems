# Movie Recommendations

This is an individual 10 days learning project at BeCode Liège.

## Introduction :

The Movie Recommendation App is a web application that recommends movies to users based on the similarity of movie overviews. The application is built using the `streamlit` library in Python, which is used to build interactive web applications. The app uses a pre-trained `SentenceTransformer` model to generate embeddings of movie overviews, and then computes the cosine similarity between these embeddings to generate a similarity matrix.

## data_preprocessor.ipynb :

This notebook loads the movie dataset from a CSV file and preprocesses it by 
dropping null values, removing movies with less than 10 votes and a vote average below 6, dropping duplicate titles with the lowest vote average, and sorting the dataframe in alphabetical order. It assigns an index to each movie and saves the preprocessed data to a new CSV file that will be used in the movie recommendation app.

## streamlit.py :

The code is divided into several sections, each of which performs a specific function. We will now go over each section of the code in more detail.

### Imports :

The first section of the code imports the necessary libraries and modules that are needed to run the application. The following libraries and modules are imported :

- `pandas` - for data manipulation and analysis
- `cosine_similarity` - from `sklearn.metrics.pairwise` for computing cosine similarity between embeddings
- `SentenceTransformer` - for generating sentence embeddings
- `streamlit` - for building the web application

### Load Movie Dataset :

The second section of the code loads the movie metadata from a CSV file called `movies_metadata_preprocessed.csv`. The metadata includes information such as the title of the movie, the overview of the movie, and other relevant information. The `pandas` library is used to load the CSV file.

### Functions :

The third section of the code defines several functions that are used throughout the application. The functions include :

- `get_title(index)` - Returns the title of a movie given its index.
- `get_index(title)` - Returns the index of a movie given its title.
- `compute_similarity_matrix()` - Computes the similarity matrix between the embeddings of the movie overviews.

### Streamlit App :

The fourth section of the code creates the streamlit application. This includes defining the title of the application, setting the appearance of the page, and creating an input field for the user to enter the name of a movie.

The `page_bg_img` variable is used to set the background image of the application. The `st.markdown()` function is used to display the background image. The `st.selectbox()` function is used to create an input field where the user can select a movie from a dropdown list.

### Recommendation Code :

The fifth section of the code performs the recommendation based on the input movie. When the user clicks the "Get Recommendations" button, the application computes the similarity between the input movie and all other movies in the dataset. The results are then sorted based on their similarity score and the top 3 movies are displayed, along with their overviews.

The `recommendations` variable is a list of tuples containing the index of the recommended movie and its similarity score. The `sorted()` function is used to sort the recommendations based on their similarity score. The `st.write()` function is used to display the top 3 recommended movies along with their overviews.

### Caching :

The code also includes caching using `@st.cache_resource` to speed up the computation of the similarity matrix. This means that the similarity matrix will only be computed once and then stored in cache, so that subsequent requests for the similarity matrix will be served from cache instead of being recomputed. This significantly reduces the time it takes to generate the recommendations.

## Installation/Requirements :

A *requirement.txt* file is provided, Python 3.10.6 was used as a base environnement.

We use the following librairies :

* pandas==1.5.2

* scikit_learn==1.2.2

* sentence_transformers==2.2.2

* streamlit==1.21.0

## Results and Conclusions :

The project was enjoyable, but we had to complete it quickly. We also tried to test sentiment analysis by analyzing the overall sentiment of a movie's subtitle file, but it took a lot of time and we didn't end up implementing it due to this reason, as well as the lack of a reliable source to download full subtitles from every movie in batches.
