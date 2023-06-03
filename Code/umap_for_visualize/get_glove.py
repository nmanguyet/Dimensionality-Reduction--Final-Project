##
import io # Path
import numpy as np

def load_word_embeddings(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') # Read file
    vocab, matrix = [], []
    for line in fin: # Each line
        tokens = line.rstrip().split(' ') # Tokenize
        vocab.append(tokens[0]) # Add vocab
        matrix.append(list(map(float, tokens[1:]))) # Add embedding
    return vocab, matrix


def get_embedding_glove():
    vocab, matrix = load_word_embeddings("glove.6B.100d.txt") # Get embedding matrix with glove have 100 dim
    embedding_matrix = np.pad(matrix, ((2,0),(0,0)), mode='constant', constant_values=0.0)  # Padding to equal length
    return vocab, embedding_matrix
