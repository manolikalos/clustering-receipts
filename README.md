# Clustering and classifying new items in receipt DataFrame

## KillBillsClusteringNB.ipynb

**This notebook goes through the research process of making the kbclustering.py script and explains the code.**

## KBClustering.py

**This script provides functions for data clustering using various techniques and associating items with clusters. It utilizes the sentence-transformers library for encoding textual data and the scikit-learn library for clustering algorithms. The script also makes use of other libraries such as pandas, numpy, matplotlib, and re. It is used to treat dataframes of receipt informations.**

### Functions

- `prepare_clustering(csv_path)`: This function reads the CSV file, cleans the data, applies KMeans and Agglomerative clustering and determines the name of each cluster using NLP.
- `preprocess_text(text)`: This function performs text preprocessing (removing digits and converting the text to lowercase).
- `associate_item_with_agglomerative_cluster(item, sample_df, model, itemdescription_embeddings, cluster_labels_agglomerative, cluster_names_dict_agglomerative)`: This function associates a new item with the existing agglomerative clusters using a handmade KNN Classifier.
- `get_random_item_from_test_set(df)`: This function fetches a random item from the dataset, excluding items from the training set.

### Example Usage

```python 

import pandas as pd
import kbclustering

# Prepare clustering
sample_df, model, itemdescription_embeddings, cluster_labels_agglomerative, cluster_names_dict_agglomerative = kbclustering.prepare_clustering('data.csv')

# Generate a random item
random_item = kbclustering.get_random_item_from_test_set(df)

# Associate the random item with clusters
sample_df = kbclustering.associate_item_with_agglomerative_cluster(item, sample_df, model, itemdescription_embeddings, cluster_labels_agglomerative, cluster_names_dict_agglomerative)

```

Note: Replace 'data.csv' with the actual path to your CSV file containing the data. To use random_item you have to load your data first with

```
df = pd.read_csv(data.csv)
```

Make sure to install the required libraries mentioned in the script and have the necessary data file before running the code.

