##
import re
import nltk # Using stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize # Tokenize text
from nltk.stem import PorterStemmer # Filler out the short word
from nltk.corpus import stopwords # Stopwords

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


def clean_df(df, text_column, clean_text_column):
    df[clean_text_column] = df[text_column].apply(lambda x: text_cleaner(x)) # data cleaning
    df['len_sentence'] = df[clean_text_column].apply(lambda x: len(x)) # Get len of text
    df = df.drop(df[df['len_sentence'] == 0].index) # Drop text don't have any word
    return df


