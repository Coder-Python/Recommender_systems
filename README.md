# Movie Recommendations

This is an independent 8-day learning project being completed at BeCode Liège.

## Introduction :

The Movie Recommendation App is a web application that recommends movies to users based on the similarity of movie overviews. The application is built using the `streamlit` library in Python, which is used to build interactive web applications. The app uses a pre-trained `SentenceTransformer` model to generate embeddings of movie overviews, and then computes the cosine similarity between these embeddings to generate a similarity matrix.

## The SentenceTransformer library :

The "sentence transformer" library is used to generate high-dimensional vector representations of the movie plot. These vector representations capture the underlying meaning of the plot and can be used to compare the similarity between the input movie and other movies in a database.

The library utilizes a pre-trained transformer-based neural network architecture to generate the vector representations. The neural network model processes the input movie plot through multiple layers of encoding, generating a high-dimensional vector representation of the plot.

Once the vector representations of movie plots are generated, they can be compared using similarity metrics to determine the movies with similar plots in the database. This enables the movie recommendation system to suggest movies with similar themes and plot elements to the input movie, improving the accuracy of the recommendation system.

Overall, the "sentence transformer" library provides a powerful tool for generating vector representations of movie plots, making it useful in movie recommendation systems based on plot similarity.

## Tool and App :

## 1. *data_preprocessor.ipynb* :

This notebook loads the movie dataset from a CSV file and preprocesses it by 
dropping null values, removing movies with less than 10 votes and a vote average below 6, dropping duplicate titles with the lowest vote average, and sorting the dataframe in alphabetical order. It assigns an index to each movie and saves the preprocessed data to a new CSV file that will be used in the movie recommendation app.

## 2. *streamlit.py* :

This is the streamlit application.

The code is divided into several sections, each of which performs a specific function. We will now go over each section of the code in more detail.

### 2.1 Imports :

The first section of the code imports the necessary libraries and modules that are needed to run the application. The following libraries and modules are imported :

- `pandas` - for data manipulation and analysis
- `cosine_similarity` - from `sklearn.metrics.pairwise` for computing cosine similarity between embeddings
- `SentenceTransformer` - for generating sentence embeddings
- `streamlit` - for building the web application

### 2.2 Load Movie Dataset :

The second section of the code loads the movie metadata from a CSV file called `movies_metadata_preprocessed.csv`. The metadata includes information such as the title of the movie, the overview of the movie, and other relevant information. The `pandas` library is used to load the CSV file.

### 2.3 Functions :

The third section of the code defines several functions that are used throughout the application. The functions include :

- `get_title(index)` - Returns the title of a movie given its index.
- `get_index(title)` - Returns the index of a movie given its title.
- `compute_similarity_matrix()` - Computes the similarity matrix between the embeddings of the movie overviews.

### 2.4 Streamlit App :

The fourth section of the code creates the streamlit application. This includes defining the title of the application, setting the appearance of the page, and creating an input field for the user to enter the name of a movie.

The `page_bg_img` variable is used to set the background image of the application and other visual details. The `st.markdown()` function is used to display the background image. The `st.selectbox()` function is used to create an input field where the user can select a movie from a dropdown list.

### 2.5 Recommendation Code :

The fifth section of the code performs the recommendation based on the input movie. When the user clicks the "Get Recommendations" button, the application computes the similarity between the input movie and all other movies in the dataset. The results are then sorted based on their similarity score and the top 3 movies are displayed, along with their overviews.

The `recommendations` variable is a list of tuples containing the index of the recommended movie and its similarity score. The `sorted()` function is used to sort the recommendations based on their similarity score. The `st.write()` function is used to display the top 3 recommended movies along with their overviews.

### 2.6 Caching :

The code also includes caching using `@st.cache_resource` to speed up the computation of the similarity matrix. This means that the similarity matrix will only be computed once and then stored in cache, so that subsequent requests for the similarity matrix will be served from cache instead of being recomputed. This significantly reduces the time it takes to generate the recommendations.

## Visuals :

Thanks to the autosuggestion feature, we can begin typing a movie title and receive a list of matching titles :

![dropdown.png](./visuals/dropdown.png)

Afterward, we can retrieve the results by clicking the "Get Recommendations" button :

![results.png](./visuals/results.png)

As we can see, the recommendations are related to the theme of artificial intelligence and robots, which is similar to the input movie "AI Artificial Intelligence 2001." The algorithm that generated the recommendations analyze the movie plot to come up with similar movies that users might enjoy.

The first recommendation, "ABE," is a short film that explores the emotional side of robots and their relationship with humans. It shares a similar theme with "AI Artificial Intelligence 2001" as both movies depict the relationship between humans and artificial beings.

The second recommendation, "Ex Machina," is a sci-fi movie that explores the concept of consciousness and artificial intelligence. Which is also very similar.

The third recommendation, "Blinky™," is a short film that explores the impact of anger and emotions on a boy and his robot. This recommendation seems to be more focused on the emotional aspects of the story and the relationships between the characters.

Overall, these recommendations are suitable for users who are interested in movies about artificial intelligence, robots, and their relationships with humans.

## Installation/Requirements/Usage :

We have provided a `requirements.txt` file, and the project was developed using Python 3.10.6. To install the required libraries, you can use the following command :

```shell
pip install -r requirements.txt
```

This project use the following librairies :

* pandas==1.5.2

* scikit_learn==1.2.2

* sentence_transformers==2.2.2

* streamlit==1.21.0

To run the app, first install the libraries listed above, and then run the app by executing this command :

```shell
streamlit run streamlit.py
```

## Results and Conclusions :

The project was enjoyable, but we had to complete it quickly. We also tried to test sentiment analysis by analyzing the overall sentiment of a movie's subtitle file, but it took a lot of time and we didn't end up implementing it due to this reason, as well as the lack of a reliable source to download full subtitles from every movie in batches.
