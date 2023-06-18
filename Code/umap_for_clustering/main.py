import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from umap import UMAP
# from preprocessor import build_preprocessing_pipeline

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
np.random.seed(42)

def build_umap_preprocessing_pipeline(
    numeric_cols,
    categorical_cols,
    text_col,
):

    def text_preprocessor(X):
        stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()

        X_transformed = []
        for text in X:
            tokens = text.lower().split()
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
            text = ' '.join(tokens)
            X_transformed.append(text)

        return np.array(X_transformed)

    function_transformer = FunctionTransformer(text_preprocessor)
    count_vectorizer = CountVectorizer(min_df=0.01, max_df=0.85)
    text_pipeline = Pipeline([
        ('function_transformer', function_transformer),
        ('count_vectorizer', count_vectorizer),
        ('svd', TruncatedSVD(n_components=100)),
    ])

    # Numeric
    standard_scaler = StandardScaler()
    imputer = SimpleImputer()
    numeric_pipeline = Pipeline([
        ('imputer', imputer),
    ])

    # Categorical
    hashing_vectorizer = HashingVectorizer(n_features=1000)
    categorical_pipeline = Pipeline([
        ('join_columns', FunctionTransformer(
            lambda X: X[high_card_cat_cols].apply(lambda x: '|'.join(x), axis=1)
        )),
        ('hashing_vectorizer', hashing_vectorizer),
        ('svd', TruncatedSVD(n_components=20)),
    ])

    return Pipeline([
        ('preprocessing', ColumnTransformer([
            ('text_pipeline', text_pipeline, text_col),
            ('numeric_pipeline', numeric_pipeline, numeric_cols),
            ('categorical_pipeline', categorical_pipeline, categorical_cols),
        ])),
        ('scale', standard_scaler),
    ])



# Data preparation
raw_df = pd.read_csv('./imdb_top_1000.csv')

numeric_cols = ['IMDB_Rating', 'Meta_score', 'No_of_Votes']
high_card_cat_cols = ['Released_Year', 'Director']
text_component_cols = ['Star1', 'Star2', 'Star3', 'Star4', 'Overview', 'Series_Title']
text_col_name = 'text'

df = raw_df[numeric_cols + high_card_cat_cols]
df[text_col_name] = raw_df[text_component_cols].apply(
    lambda row: ' '.join(row.values.astype(str)), axis=1
)
df[text_col_name] = raw_df[text_component_cols].astype(str).agg(' '.join, axis=1)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(raw_df['Genre'].apply(lambda x: x.split(', ')[0]))


# Preprocessing
pipeline = build_umap_preprocessing_pipeline(numeric_cols, high_card_cat_cols, text_col_name)
pipeline.fit(df)
X = pipeline.transform(df)
y = df['label'].values

idx_leftout = 10
x_leftout = X[idx_leftout]
label_leftout = y[idx_leftout]
print(label_encoder.inverse_transform([label_leftout]).item())
X = np.delete(X, idx_leftout, axis=0)
y = np.delete(y, idx_leftout, axis=0)


# UMAP
umap = UMAP(n_components=2, n_neighbors=25, random_state=42)
projections = umap.fit_transform(X, y=y)

df_plot = pd.DataFrame({'x': projections[:, 0], 'y': projections[:, 1], 'color': y})
x_leftout_projected = umap.transform(x_leftout.reshape(1, -1))

group_means = df_plot.groupby('color').agg({'x': 'mean', 'y': 'mean'})
texts = []
for idx in group_means.index:
    x, y = group_means.loc[idx].x, group_means.loc[idx].y
    label = label_encoder.inverse_transform([idx]).item()
    texts.append([x, y, label])


plt.figure()
plt.scatter(*df_plot[['x', 'y']].values.T, c=df_plot['color'])
plt.scatter(*x_leftout_projected.T, s=100, c='red')

for x, y, label in texts:
    plt.text(x, y, label, c='violet')


##
pd.DataFrame({
    'Genre': raw_df['Genre'],
    'count': raw_df['Genre'].apply(lambda x: len(x.split(', ')))
}).sort_values('count').head(40)


