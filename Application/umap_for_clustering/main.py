##
from get_data import load_data
from pre_process import select_column, text_cleaner
from get_glove import get_embedding_glove
from get_embed_from_glove import get_index_embedding, get_embedding_matrix
from sklearn.manifold import TSNE # T-sne algorithm
from sklearn.metrics import accuracy_score # Check algorithm is good or not
from sklearn.metrics import silhouette_score # Evaluate result
from sklearn.cluster import KMeans # Cluster data after embedding
import pandas as pd # Datafram
import numpy as np # Array and math
import matplotlib.pylab as plt # Plot
import umap  # pip install umap-learn 

##
def concatenate_text(data):
    return data['Series_Title'] + data['Genre'] + data['Overview'] + data['Director']


def concat_data(col_1, col_2):
    concat_data = np.hstack((col_1, col_2))
    return concat_data


def umap_reduce_dim(X):
    manifold = umap.UMAP(n_neighbors=30,
                        min_dist=0.1,
                        n_components=2,
                        metric='cosine',
                        random_state=62
                        )
    X_reduced = manifold.fit_transform(X)
    return X_reduced


def t_sne_reduce_dim(X):
    tsne = TSNE(n_components=2, perplexity=30, verbose=1, random_state=123)
    X_reduced = tsne.fit_transform(np.array(X))
    return X_reduced


def k_means(X):
    kmeans = KMeans(n_clusters=20, random_state=62, n_init="auto").fit(X)
    label_kmeans = kmeans.labels_
    return label_kmeans


def plot_scatter(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=0.5, cmap='Spectral')
    plt.show()


def evaluation(X, y_pred, y_not_reduce):
    acc = accuracy_score(y_not_reduce, y_pred)
    score_sil = silhouette_score(X, y_not_reduce)
    return acc, score_sil

##
# Load data
path = './imdb_top_1000.csv'
df = load_data(path)
text_data = concatenate_text(df)
X = select_column(df, text_data)
tokenized_texts = X['text'].to_list()


##
# Get embedding from glove
vocab, embedding_matrix = get_embedding_glove() # Get embedding matrix from glove
X_index = get_index_embedding(tokenized_texts, vocab, embedding_matrix)
text_embedding = get_embedding_matrix(X_index, embedding_matrix)


##
# Get other feature and concat all
other_feature = np.array(X.drop('text', axis=1)).reshape(1000, -1)
concatenated_data = concat_data(text_embedding, other_feature)


##
# Reduce dimensionality and using K-means
kmeans = KMeans(n_clusters=20, random_state=62, n_init="auto").fit(concatenated_data)
label_only_kmeans = kmeans.labels_
X_reduced_tsne = t_sne_reduce_dim(concatenated_data)
X_reduced_umap = umap_reduce_dim(concatenated_data)
label_kmeans_umap = k_means(X_reduced_umap)
label_kmeans_tnse = k_means(X_reduced_tsne)
plot_scatter(X_reduced_tsne, label_kmeans_tnse)
plot_scatter(X_reduced_umap, label_kmeans_umap)


##
# Evaluation of umap
acc_tsne, score_tsne = evaluation(X_reduced_tsne, label_kmeans_tnse, label_only_kmeans)
print("Ratio of difference between dimensionality reduction and non-dimension of t-sne: ", acc_tsne)
print("Score silhouette of t-sne:", score_tsne)

##
# Evaluation of tnse
acc_umap, score_umap = evaluation(X_reduced_umap, label_kmeans_umap, label_only_kmeans)
print("Ratio of difference between dimensionality reduction and non-dimension of umap: ", acc_umap)
print("Score silhouette of umap:", score_umap)
