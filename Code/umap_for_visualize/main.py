##
from get_data import load_data
from clean_data import clean_df, text_cleaner
from get_glove import get_embedding_glove
from bag_of_word import bag_of_word_embedding
from tfidf import tf_idf_embedding
from get_embed_from_glove import get_index_embedding, get_embedding_matrix
from lstm import lstm_embedding
from sklearn.model_selection import train_test_split # Split train - test
from sklearn.metrics import silhouette_score
from umap import validation # Trustworthiness_umap
from warnings import filterwarnings # Delete warning
import umap # Umap algorithm
import matplotlib.pyplot as plt # Plot

##
def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                       random_state=42)
    return X_train, X_test, y_train, y_test


def umap_reduce_dim(X_train, y_train, X_test):
    manifold = umap.UMAP(n_neighbors=30,
                        min_dist=0.1,
                        n_components=2,
                        metric='cosine',
                        random_state=42
                        )
    X_reduced_train = manifold.fit_transform(X_train, y_train)
    X_reduced_test = manifold.transform(X_test)
    return X_reduced_train, X_reduced_test


def plot_scatter(X_train, y_train, X_test, y_test, algorithm):
    print("Algorithm: " + algorithm)
    print("Data train after embedding")
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=0.5, cmap='Spectral')
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=0.5, cmap='Spectral')
    print("Data test after embedding")
    plt.show()


##
# Initial
text_column = 'text'
clean_text_column = 'clean_text_stem'
target_column = 'source'
n_class = 20


##
# Load data and split train - test
df, dataset = load_data()
df_clean = clean_df(df, text_column, clean_text_column)
X_train, X_test, y_train, y_test = split_train_test(df_clean[text_column], df_clean[target_column]) # Split train test


##
# Bag of word
X_train_BOW, X_test_BOW = bag_of_word_embedding(X_train, X_test)
X_reduced_train_BOW, X_reduced_test_BOW = umap_reduce_dim(X_train_BOW.toarray(), y_train,
                                                          X_test_BOW.toarray())
plot_scatter(X_reduced_train_BOW, y_train, X_reduced_test_BOW, y_test, 'bag of word')


##
# Tfidf
X_train_tfidf, X_test_tfidf = tf_idf_embedding(X_train, X_test) 
X_reduced_train_tfidf, X_reduced_test_tfidf = umap_reduce_dim(X_train_tfidf.toarray(), y_train,
                                                              X_test_tfidf.toarray())
plot_scatter(X_reduced_train_tfidf, y_train, X_reduced_test_tfidf, y_test, 'tf-idf')



##
# Pre-mode glove
X_train_glove = X_train.apply(lambda x: text_cleaner(x)) # data cleaning
X_test_glove = X_test.apply(lambda x: text_cleaner(x)) # data cleaning
tokenized_texts_train = X_train_glove.to_list()
tokenized_texts_test = X_test_glove.to_list()
vocab, embedding_matrix = get_embedding_glove() # Get embedding matrix from glove
X_index_train, y_label_train = get_index_embedding(tokenized_texts_train, y_train, vocab)
X_index_test, y_label_test = get_index_embedding(tokenized_texts_test, y_test, vocab)
text_embedding_train_glove = get_embedding_matrix(X_index_train, embedding_matrix)
text_embedding_test_glove = get_embedding_matrix(X_index_test, embedding_matrix)
X_reduced_train_glove, X_reduced_test_glove = umap_reduce_dim(text_embedding_train_glove, y_train,
                                                              text_embedding_test_glove)
plot_scatter(X_reduced_train_glove, y_label_train, X_reduced_test_glove, y_label_test, 'pre-train model')


##
# LSTM
text_embedding_train_lstm, target_matrix_train, text_embedding_test_lstm, target_matrix_test = lstm_embedding(X_index_train, y_label_train, X_index_test, y_label_test, embedding_matrix, n_class)
X_reduced_train_lstm, X_reduced_test_lstm = umap_reduce_dim(text_embedding_train_lstm,
                                                            target_matrix_train,
                                                            text_embedding_test_lstm)
plot_scatter(X_reduced_train_lstm, target_matrix_train, X_reduced_test_lstm, target_matrix_test, 'lstm')


##
# Evaluation
score_BOW = silhouette_score(X_reduced_test_BOW, y_test)
print("Silhouette score of bag of word is ", score_BOW)
score_tdidf = silhouette_score(X_reduced_test_tfidf, y_test)
print("Silhouette score of tf-idf is ", score_tdidf)
score_glove = silhouette_score(X_reduced_test_glove, y_test)
print("Silhouette score of pre-train model glove is ", score_glove)
score_lstm = silhouette_score(X_reduced_test_lstm, y_test)
print("Silhouette score of tf-idf is ", score_lstm)

##
K = 30
filterwarnings('ignore')

trustworthiness_bow = validation.trustworthiness_vector(source=X_train_BOW.toarray()
                                  ,embedding=X_reduced_train_BOW
                                  ,max_k=K)
trustworthiness_tfidf = validation.trustworthiness_vector(source=X_train_BOW.toarray()
                                  ,embedding=X_reduced_train_tfidf
                                  ,max_k=K)
trustworthiness_glove = validation.trustworthiness_vector(source=text_embedding_train
                                  ,embedding=X_reduced_train_glove
                                  ,max_k=K)

trustworthiness_lstm = validation.trustworthiness_vector(source=np.array(embedding_matrix_lstm_train)
                                  ,embedding=X_reduced_train_lstm
                                  ,max_k=K)

filterwarnings('default')


plt.plot(trustworthiness_bow)
plt.plot(trustworthiness_tfidf)
plt.plot(trustworthiness_glove)
plt.plot(trustworthiness_lstm)
plt.ylabel("Trustworthiness score")
plt.xlabel("Value of K")
plt.title(f"Trustworthiness at {K}")
plt.legend(["bow", "tfdif", "glove", "lstm"], loc="upper right")
