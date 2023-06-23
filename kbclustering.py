import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import re
import random

def prepare_clustering(csv_path):
    """
    Executes both a kmeans and an agglomerative clustering of a sample of the given data set.
    """
    # Load the dataset from a CSV file, and parse certain columns as date type
    df = pd.read_csv(csv_path, parse_dates=['date', 'createdAt', 'updatedAt'])

    # Define the list of columns to be dropped from the dataset
    columns_to_drop = ['Unnamed: 0', 'parent', 'taxDescription', 'storeId', 'createdAt', 'updatedAt']
    
    # Drop the unwanted columns from the dataset
    df = df.drop(columns=columns_to_drop)

    # Remove rows with missing values in specific columns ('amount', 'taxRate', 'itemName','description')
    df.dropna(subset=['amount', 'taxRate', 'itemName','description'], inplace=True)

    # Combine 'itemName' and 'description' into a new 'itemdescription' column and remove any numeric characters
    df['itemdescription'] = df['itemName'] + ' ' + df['description']
    df['itemdescription'] = df['itemdescription'].apply(lambda x: re.sub(r'\d+', '', x))
    
    # Scale the 'amount' and 'taxRate' columns using StandardScaler for normalization
    scaler = StandardScaler()
    df['amount_scaled'] = scaler.fit_transform(df[['amount']])
    df['taxRate_scaled'] = scaler.fit_transform(df[['taxRate']])

    # Randomly sample 20000 records from the original dataframe, for efficient processing
    sample_df = df.sample(n=20000, random_state=42)
    
    # Use a pre-trained multilingual sentence transformer model to convert item descriptions into embeddings (numeric vector representations)
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    itemdescription_embeddings = model.encode(sample_df['itemdescription'].tolist(), convert_to_tensor=True)

    # Concatenate the item description embeddings with scaled tax rate and amount features to form the complete feature set for clustering
    features = np.concatenate((itemdescription_embeddings, sample_df[['taxRate_scaled', 'amount_scaled']].values), axis=1)

    # Initialize KMeans clustering model with 4 clusters and a fixed random state for reproducibility
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(features)
    # Retrieve the cluster labels assigned by the KMeans model
    cluster_labels_kmeans = kmeans.labels_
    
    # Agglomerative clustering model with 8 clusters
    agglomerative = AgglomerativeClustering(n_clusters=8)
    agglomerative.fit(itemdescription_embeddings)
    # Retrieve the cluster labels for each sample
    cluster_labels_agglomerative = agglomerative.labels_

    sample_df['kmeans_cluster'] = cluster_labels_kmeans

    sample_df['agglomerative_cluster'] = cluster_labels_agglomerative

    # Transform item descriptions into TF-IDF format
    vectorizer = TfidfVectorizer(smooth_idf=True)
    item_descriptions_tfidf = vectorizer.fit_transform(sample_df['itemdescription'])
    
    # Run Latent Dirichlet Allocation
    n_topics = 4
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(item_descriptions_tfidf)
    
    # Retrieve all feature names from TF-IDF vectorizer
    feature_names = vectorizer.get_feature_names_out()
    n_top_words = 1  # Number of top words to display

    cluster_names_kmeans = []
    
    # Determine cluster names by selecting the highest TF-IDF scoring term for each cluster
    for cluster_id in range(n_topics):
        cluster_items = sample_df[sample_df['kmeans_cluster'] == cluster_id]
        cluster_descriptions = ' '.join(cluster_items['itemdescription'])
        cluster_descriptions_tfidf = vectorizer.transform([cluster_descriptions])
        cluster_term_scores = cluster_descriptions_tfidf.toarray()[0]
        top_words_indices = cluster_term_scores.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        cluster_name = ", ".join(top_words)
        cluster_names_kmeans.append(cluster_name)

      
    # TF-IDF transformation of item descriptions
    vectorizer = TfidfVectorizer(smooth_idf=True)
    item_descriptions_tfidf = vectorizer.fit_transform(sample_df['itemdescription'])
    n_topics = 8
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(item_descriptions_tfidf)
    feature_names = vectorizer.get_feature_names_out()
    n_top_words = 1

    cluster_names_agglomerative = []
    
    # Determine cluster names by selecting the highest TF-IDF scoring term for each cluster
    for cluster_id in range(n_topics):
        cluster_items = sample_df[sample_df['agglomerative_cluster'] == cluster_id]
        cluster_descriptions = ' '.join(cluster_items['itemdescription'])
        cluster_descriptions_tfidf = vectorizer.transform([cluster_descriptions])
        cluster_term_scores = cluster_descriptions_tfidf.toarray()[0]
        top_words_indices = cluster_term_scores.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        cluster_name = ", ".join(top_words)
        cluster_names_agglomerative.append(cluster_name)

    # Assign the labels resulting from K-means clustering to a new column in the sample_df DataFrame
    sample_df['kmeans_cluster'] = cluster_labels_kmeans
    
    # Create a dictionary that maps each cluster label to its corresponding name (for K-means)
    cluster_names_dict_kmeans = {label: name for label, name in zip(range(n_topics), cluster_names_kmeans)}

    # Map the K-means cluster labels in the DataFrame to their corresponding names
    sample_df['kmeans_cluster_name'] = sample_df['kmeans_cluster'].map(cluster_names_dict_kmeans)

    # Assign the labels resulting from agglomerative clustering to a new column in the sample_df DataFrame
    sample_df['agglomerative_cluster'] = cluster_labels_agglomerative

    # Create a dictionary that maps each cluster label to its corresponding name (for agglomerative clustering)
    cluster_names_dict_agglomerative = {label: name for label, name in zip(range(len(cluster_names_agglomerative)), cluster_names_agglomerative)}

    # Map the agglomerative cluster labels in the DataFrame to their corresponding names
    sample_df['agglomerative_cluster_name'] = sample_df['agglomerative_cluster'].map(cluster_names_dict_agglomerative)

    return sample_df, model, itemdescription_embeddings, cluster_labels_agglomerative, cluster_names_dict_agglomerative
    
def preprocess_text(text):
    """
    Preprocesses the given text by removing digits, leading and trailing whitespaces, and converting to lowercase.
    
    Parameters:
    text (str): The text string to be preprocessed.
    
    Returns:
    text (str): The preprocessed text.
    """
    
    # Remove any digits from the text
    text = re.sub(r'\d+', '', text)
    
    # Remove leading and trailing whitespaces
    text = text.strip()
    
    # Convert text to lowercase
    text = text.lower()
    
    return text

def associate_item_with_agglomerative_cluster(item, sample_df, model, itemdescription_embeddings, cluster_labels_agglomerative, cluster_names_dict_agglomerative):
    """Associates a new item with the existing agglomerative clusters using and handmade KNN Classifier."""
    
    # Preprocess and encode the text 
    item['itemdescription'] = item['itemName'] + ' ' + item['description']
    item['itemdescription'] = re.sub(r'\d+', '', item['itemdescription'])
    description = preprocess_text(item['itemdescription'])
    item_embedding = model.encode([description], convert_to_tensor=True)
    
    # Use distances to find the index of the closest item in the dataset
    distances = pairwise_distances(item_embedding, itemdescription_embeddings)
    closest_index = np.argmin(distances)
    agglomerative_cluster_label = cluster_labels_agglomerative[closest_index]
    
    # Retrieve the cluster informations
    item['agglomerative_cluster'] = agglomerative_cluster_label
    agglomerative_cluster_name = cluster_names_dict_agglomerative[agglomerative_cluster_label]
    item['agglomerative_cluster_name'] = agglomerative_cluster_name
    
    # Add the item to the DataFrame
    sample_df = sample_df.append(item, ignore_index=True)
    print("Item:")
    print(item)
    return agglomerative_cluster_label, agglomerative_cluster_name

def get_random_item_from_test_set(df):
    """Fetches a random item from the dataset, excluding items from the training set."""
    
    # Exclude items that are already in sample_df
    excluded_indices = sample_df.index
    eligible_indices = df.index[~df.index.isin(excluded_indices)]
    
    # Select a random index from the eligible indices
    random_index = random.choice(eligible_indices)
    
    # Get the random item from df
    random_item = df.loc[random_index]
    
    # Print the item details
    print("Random Item Details:")
    print("ID: ", random_item['id'])
    print("Name: ", random_item['itemName'])
    print("Description: ", random_item['description'])
    print("Amount: ", random_item['amount'])
    print("Tax Rate: ", random_item['taxRate'])
    print()
    
    return random_item
