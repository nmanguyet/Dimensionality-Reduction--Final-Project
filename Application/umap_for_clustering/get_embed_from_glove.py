##
import numpy as np

def text_to_index(tokenized_texts, vocab, unknown_word):
    X = [] # Contain index of all text
    word_to_index = {word: index+2 for index, word in enumerate(vocab)}

    for text in tokenized_texts: # For each text
        cur_text_indices = [] # Contain index of a text
        for word in text: # For each word
            if word in word_to_index:
                cur_text_indices.append(word_to_index[word])
            else:
                cur_text_indices.append(unknown_word)
        X.append(cur_text_indices) # Append index of text
    return X


def get_index_embedding(tokenized_texts, vocab, matrix):
    X = text_to_index(tokenized_texts, vocab, 1) # Convert our text to index using above embedding matrix
    return X


def get_embedding_matrix(X, embedding_matrix):
    text_embedding = [] 
    for text_index in X: # Each text
        embeddings = []
        for word_index in text_index: # Each word
            embeddings.append(embedding_matrix[word_index]) # Get embedding vector from pre-train model
        text_embedding.append(np.mean(embeddings, axis=0)) # Using mean for all vector of text
    return np.array(text_embedding)
