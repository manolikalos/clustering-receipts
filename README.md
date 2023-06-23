# Clustering and classifying new items in receipt DataFrame

## KillBillsClusteringNB.ipynb

**This notebook goes through the research process of making the kbclustering.py script and explains the code.**

## KBClustering.py

**This script provides functions for data clustering using various techniques and associating items with clusters. It utilizes the sentence-transformers library for encoding textual data and the scikit-learn library for clustering algorithms. The script also makes use of other libraries such as pandas, numpy, matplotlib, and re. It is used to treat dataframes of receipt informations.**

### Functions

**1. prepare_clustering(csv_path)**

This function prepares the data and executes clusterings using both kmeans and agglomerative methods.


Parameters:

- csv_path: The file path to the CSV file containing the data.

Returns:

- sample_df: A DataFrame containing the sampled data with added cluster labels and cluster names.
- model: The SentenceTransformer model used for encoding text.
- kmeans: The trained KMeans clustering model.
- agglomerative: The trained AgglomerativeClustering model.
- cluster_names_dict_kmeans: A dictionary mapping cluster labels to their corresponding names for K-means clustering.
- cluster_names_dict: A dictionary mapping cluster labels to their corresponding names for Agglomerative clustering.

**2. preprocess_text(text)**

This function preprocesses the given text by removing numerical digits, stripping leading/trailing whitespace, and converting it to lowercase.
It will be used in the associate_item_with_cluster function.

Parameters:

- text: The text to be preprocessed.

Returns:

- The preprocessed text.

**3. associate_item_with_cluster(item, sample_df, model, kmeans, agglomerative)**

This function associates a new item with clusters, and add the item to the DataFrame.

Parameters:

- item: A dictionary representing the item with the following keys: 'description', 'amount', 'taxRate'.
- sample_df: The DataFrame containing the sampled data with existing cluster labels.
- model: The SentenceTransformer model used for encoding text.
- kmeans: The trained KMeans clustering model.
- agglomerative: The trained AgglomerativeClustering model.

Returns:

- agglomerative_cluster_label, agglomerative_cluster_name : Cluster label and name of the given item.

**4. generate_random_item(df)**

This function generates a random item from the given DataFrame.

Parameters:

- df: The DataFrame containing the data.

Returns:

- The randomly selected item as a dictionary.

### Example Usage

```python 

import pandas as pd
import kbclustering

# Prepare clustering
sample_df, model, kmeans, agglomerative, cluster_names_kmeans, cluster_names_agglomerative = clustering.prepare_clustering('data.csv')

# Generate a random item
random_item = clustering.generate_random_item(sample_df)

# Associate the random item with clusters
sample_df = clustering.associate_item_with_cluster(random_item, sample_df, model, kmeans, agglomerative)

```

Note: Replace 'data.csv' with the actual path to your CSV file containing the data.

Make sure to install the required libraries mentioned in the script and have the necessary data file before running the code.

