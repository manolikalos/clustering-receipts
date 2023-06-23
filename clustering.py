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

df = pd.read_csv('killbillsdata.csv', parse_dates=['date', 'createdAt', 'updatedAt'])

columns_to_drop = ['Unnamed: 0', 'parent', 'taxDescription', 'storeId', 'createdAt', 'updatedAt']
df = df.drop(columns=columns_to_drop)

df.dropna(subset=['amount', 'taxRate', 'itemName','description'], inplace=True)

df['itemdescription'] = df['itemName'] + ' ' + df['description']
df['itemdescription'] = df['itemdescription'].apply(lambda x: re.sub(r'\d+', '', x))

scaler = StandardScaler()
df['amount_scaled'] = scaler.fit_transform(df[['amount']])
df['taxRate_scaled'] = scaler.fit_transform(df[['taxRate']])

sample_df = df.sample(n=20000, random_state=42)

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
itemdescription_embeddings = model.encode(sample_df['itemdescription'].tolist(), convert_to_tensor=True)

features = np.concatenate((itemdescription_embeddings, sample_df[['taxRate_scaled', 'amount_scaled']].values), axis=1)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(features)
cluster_labels_kmeans = kmeans.labels_

agglomerative = AgglomerativeClustering(n_clusters=8)
agglomerative.fit(itemdescription_embeddings)
cluster_labels_agglomerative = agglomerative.labels_

tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(itemdescription_embeddings)

sample_df['kmeans_cluster'] = cluster_labels_kmeans

sample_df['agglomerative_cluster'] = cluster_labels_agglomerative

vectorizer = TfidfVectorizer(smooth_idf=True)
item_descriptions_tfidf = vectorizer.fit_transform(sample_df['itemdescription'])
n_topics = 4
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(item_descriptions_tfidf)
feature_names = vectorizer.get_feature_names_out()
n_top_words = 1  # Number of top words to display

cluster_names_kmeans = []
for cluster_id in range(n_topics):
    cluster_items = sample_df[sample_df['kmeans_cluster'] == cluster_id]
    cluster_descriptions = ' '.join(cluster_items['itemdescription'])
    cluster_descriptions_tfidf = vectorizer.transform([cluster_descriptions])
    cluster_term_scores = cluster_descriptions_tfidf.toarray()[0]
    top_words_indices = cluster_term_scores.argsort()[:-n_top_words - 1:-1]
    top_words = [feature_names[i] for i in top_words_indices]
    cluster_name = ", ".join(top_words)
    cluster_names_kmeans.append(cluster_name)

vectorizer = TfidfVectorizer(smooth_idf=True)
item_descriptions_tfidf = vectorizer.fit_transform(sample_df['itemdescription'])
n_topics = 8
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(item_descriptions_tfidf)
feature_names = vectorizer.get_feature_names_out()
n_top_words = 1

cluster_names_agglomerative = []
for cluster_id in range(n_topics):
    cluster_items = sample_df[sample_df['agglomerative_cluster'] == cluster_id]
    cluster_descriptions = ' '.join(cluster_items['itemdescription'])
    cluster_descriptions_tfidf = vectorizer.transform([cluster_descriptions])
    cluster_term_scores = cluster_descriptions_tfidf.toarray()[0]
    top_words_indices = cluster_term_scores.argsort()[:-n_top_words - 1:-1]
    top_words = [feature_names[i] for i in top_words_indices]
    cluster_name = ", ".join(top_words)
    cluster_names_agglomerative.append(cluster_name)

sample_df['kmeans_cluster'] = cluster_labels_kmeans
cluster_names_dict_kmeans = {label: name for label, name in zip(range(n_topics), cluster_names_kmeans)}
sample_df['kmeans_cluster_name'] = sample_df['kmeans_cluster'].map(cluster_names_dict_kmeans)

sample_df['agglomerative_cluster'] = cluster_labels_agglomerative
cluster_names_dict = {label: name for label, name in zip(range(len(cluster_names_agglomerative)), cluster_names_agglomerative)}
sample_df['agglomerative_cluster_name'] = sample_df['agglomerative_cluster'].map(cluster_names_dict)

np.save('cluster_labels_kmeans.npy', cluster_labels_kmeans)

np.save('cluster_names_kmeans.npy', cluster_names_kmeans)

np.save('vectorizer_vocabulary.npy', vectorizer.get_feature_names_out())

np.save('cluster_labels_agglomerative.npy', cluster_labels_agglomerative)

np.save('cluster_names_agglomerative.npy', cluster_names_agglomerative)
