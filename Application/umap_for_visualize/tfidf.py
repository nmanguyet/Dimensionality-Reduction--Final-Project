##
from clean_data import eleminate_not_word
from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf_embedding(X_train, X_test):
    X_train_clean = X_train.apply(lambda x: eleminate_not_word(x))
    X_test_clean = X_test.apply(lambda x: eleminate_not_word(x))

    vectorizer_tfidf = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer_tfidf.fit_transform(X_train_clean)
    X_test_tfidf = vectorizer_tfidf.transform(X_test_clean)
    return X_train_tfidf, X_test_tfidf

