##
import pandas as pd
def load_data(path):
    df = pd.read_csv(path) # Read data from path
    return df
