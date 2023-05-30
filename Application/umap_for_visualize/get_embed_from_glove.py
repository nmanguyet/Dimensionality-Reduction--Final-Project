##
import numpy as np


def text_to_index(tokenized_texts, vocab, unknown_word):
    X = []
    word_to_index = {word: index+2 for index, word in enumerate(vocab)}

    for text in tokenized_texts:
        cur_text_indices = []
        for word in text:
            if word in word_to_index:
                cur_text_indices.append(word_to_index[word])
            else:
                cur_text_indices.append(unknown_word)
        X.append(cur_text_indices)
    return X


def get_index_embedding(tokenized_texts, target_column, vocab):
    X = text_to_index(tokenized_texts, vocab, 1) # Convert our text to index using above embedding matrix
    y = target_column.values # Get target
    return X, y


def get_embedding_matrix(X, embedding_matrix):
    text_embedding = []
    for text_index in X: # Each text
        embeddings = []
        for word_index in text_index: # Each word
            embeddings.append(embedding_matrix[word_index]) # Get embedding vector from pre-train model
        text_embedding.append(np.mean(embeddings, axis=0)) # Using mean for all vector of text
    return text_embedding
