##
from clean_data import eleminate_not_word
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_word_embedding(X_train, X_test):
    # Clean data
    X_train_clean = X_train.apply(lambda x: eleminate_not_word(x))
    X_test_clean = X_test.apply(lambda x: eleminate_not_word(x))
    # Embed data
    vectorizer_BOW = CountVectorizer(max_features=1000, stop_words='english', lowercase=False)
    X_train_BOW = vectorizer_BOW.fit_transform(X_train_clean)
    X_test_BOW = vectorizer_BOW.transform(X_test_clean)
    return X_train_BOW, X_test_BOW
