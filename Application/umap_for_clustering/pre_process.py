##
import pandas as pd

import nltk # Using stopwords
nltk.download('stopwords')
nltk.download('punkt')
import time # Measure time
from nltk.tokenize import word_tokenize # Tokenize text
from nltk.stem import PorterStemmer # Filler out the short word
from nltk.corpus import stopwords # Stopwords
import re # Replace string to clean data

def select_column(df, text_data):
    X = pd.DataFrame()
    X['text'] = text_data.apply(lambda x: text_cleaner(x))
    X['IMDB_Rating'] = df['IMDB_Rating']
    X['No_of_Votes'] = df['No_of_Votes']
    X['Released_Year'] = df['Released_Year'].replace('PG', '1943').astype(int)
    X['Runtime'] = df['Runtime'].apply(lambda x: int(x[:-4]))
    return X


def eleminate_not_word(text):
    cleaner = re.sub(r"[^a-zA-Z ]+", ' ', text.lower()) # Lowercase and strip everything except words
    return cleaner


def tokenize(text):
    cleaner = word_tokenize(text) # Tokenize text to words
    return cleaner


def eleminate_stop_word_and_padding(text):
    stopWords = set(stopwords.words('english')) # Download stopwords English from nltk
    ps = PorterStemmer() # Initial filler out short words
    clean = []
    for w in text:
        # filter out stopwords
        if w not in stopWords:
            # filter out short words
            if len(w) > 2:
                # Stem 
                clean.append(ps.stem(w))
    return clean


def text_cleaner(text):
    only_word = eleminate_not_word(text)
    token_word = tokenize(only_word)
    clean_text = eleminate_stop_word_and_padding(token_word)
    return clean_text
