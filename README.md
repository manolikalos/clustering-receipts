# KBClustering.py

This script provides functions for data clustering using various techniques and associating items with clusters. It utilizes the sentence-transformers library for encoding textual data and the scikit-learn library for clustering algorithms. The script also makes use of other libraries such as pandas, numpy, matplotlib, and re.
Functions
1. prepare_clustering(csv_path)

This function prepares the data for clustering by performing the following steps:

    Reads the data from a CSV file specified by csv_path and stores it in a pandas DataFrame.
    Drops unnecessary columns from the DataFrame.
    Removes rows with missing values in specific columns.
    Combines the 'itemName' and 'description' columns into a new 'itemdescription' column, removing any numerical digits.
    Scales the 'amount' and 'taxRate' columns using standard scaling.
    Randomly samples 20,000 rows from the DataFrame.
    Encodes the 'itemdescription' column using the 'distiluse-base-multilingual-cased-v1' sentence transformer model.
    Combines the sentence embeddings with the scaled 'amount' and 'taxRate' columns to create the feature vectors.
    Performs K-means clustering with 4 clusters on the feature vectors.
    Performs Agglomerative clustering with 8 clusters on the sentence embeddings.
    Performs t-SNE dimensionality reduction on the sentence embeddings.
    Adds the cluster labels to the sampled DataFrame.
    Performs TF-IDF vectorization on the 'itemdescription' column.
    Applies Latent Dirichlet Allocation (LDA) topic modeling with 4 topics on the TF-IDF vectors.
    Extracts the top word for each cluster from the LDA model for K-means and Agglomerative clustering.

Parameters:

    csv_path: The file path to the CSV file containing the data.

Returns:

    sample_df: A DataFrame containing the sampled data with added cluster labels and cluster names.
    model: The SentenceTransformer model used for encoding text.
    kmeans: The trained KMeans clustering model.
    agglomerative: The trained AgglomerativeClustering model.
    cluster_names_dict_kmeans: A dictionary mapping cluster labels to their corresponding names for K-means clustering.
    cluster_names_dict: A dictionary mapping cluster labels to their corresponding names for Agglomerative clustering.

2. preprocess_text(text)

This function preprocesses the given text by removing numerical digits, stripping leading/trailing whitespace, and converting it to lowercase.

Parameters:

    text: The text to be preprocessed.

Returns:

    The preprocessed text.

3. associate_item_with_cluster(item, sample_df, model, kmeans, agglomerative)

This function associates a new item with clusters by performing the following steps:

    Preprocesses the item's description using the preprocess_text function.
    Encodes the preprocessed description using the SentenceTransformer model.
    Combines the item's embedding with the existing item embeddings from the sample_df.
    Performs K-means clustering on the updated feature vectors using the provided KMeans model.
    Performs Agglomerative clustering on the updated embeddings using the provided AgglomerativeClustering model.
    Retrieves the cluster labels and names for the item from the clustering results.
    Updates the item's cluster labels and names in the sample_df.
    Prints the item details and associated clusters.

Parameters:

    item: A dictionary representing the item with the following keys: 'description', 'amount', 'taxRate'.
    sample_df: The DataFrame containing the sampled data with existing cluster labels.
    model: The SentenceTransformer model used for encoding text.
    kmeans: The trained KMeans clustering model.
    agglomerative: The trained AgglomerativeClustering model.

Returns:

    sample_df: The updated DataFrame with the added item and updated cluster labels.

4. generate_random_item(df)

This function generates a random item from the given DataFrame.

Parameters:

    df: The DataFrame containing the data.

Returns:

    The randomly selected item as a dictionary.

Example Usage

python

import pandas as pd
import kbclustering

# Prepare clustering
sample_df, model, kmeans, agglomerative, cluster_names_kmeans, cluster_names_agglomerative = clustering.prepare_clustering('data.csv')

# Generate a random item
random_item = clustering.generate_random_item(sample_df)

# Associate the random item with clusters
sample_df = clustering.associate_item_with_cluster(random_item, sample_df, model, kmeans, agglomerative)

Note: Replace 'data.csv' with the actual path to your CSV file containing the data.

Make sure to install the required libraries mentioned in the script and have the necessary data file before running the code.
