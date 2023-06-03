##
import pandas as pd # Dataframe
from sklearn.datasets import fetch_20newsgroups # Dataset

def load_data():
    # Load dataset 20newsgroups from sklearn
    dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'),
                                 shuffle=True, random_state=42)
    # Convert data to dataframe
    df = pd.DataFrame()
    df['text'] = dataset.data
    df['source'] = dataset.target
    return df, dataset
