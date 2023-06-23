# Clustering and classifying new items in receipt DataFrame

## KillBillsClusteringNB.ipynb

**This notebook goes through the research process of making the kbclustering.py script and explains the code.**

## KBClustering.py

**This script provides functions for data clustering using various techniques and associating items with clusters. It utilizes the sentence-transformers library for encoding textual data and the scikit-learn library for clustering algorithms. The script also makes use of other libraries such as pandas, numpy, matplotlib, and re. It is used to treat dataframes of receipt informations.**

### Functions

    - prepare_clustering(csv_path): This function reads the CSV file, cleans the data, applies KMeans and Agglomerative clustering, assigns each record to a cluster, and determines the name of each cluster based on the most common words in that cluster.

    - preprocess_text(text): This function performs text preprocessing such as removing digits, stripping leading and trailing whitespaces, and converting the text to lowercase.

    - associate_item_with_agglomerative_cluster(item, sample_df, model, itemdescription_embeddings, cluster_labels_agglomerative, cluster_names_dict_agglomerative): This function associates a new item with the existing agglomerative clusters using a handmade KNN Classifier.

    - get_random_item_from_test_set(df): This function fetches a random item from the dataset, excluding items from the training set.

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

