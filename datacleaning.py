import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random, pickle, ast

df = pd.read_csv("data/ecommerce_data.csv", encoding ='latin-1')

#cleaning
df['description_cleaned'] = df['description'].replace(r'[^a-zA-Z0-9\s]', '', regex=True)
df['name_cleaned'] = df['name'].replace(r'[^a-zA-Z0-9\s]', '', regex=True)
# Combine description, occasion, and age group into a single text for TF-IDF Vectorizer
df['Combined'] = df['description_cleaned'] + df['name_cleaned']

def string_to_list(string):
    return ast.literal_eval(string)

def get_first_value(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return lst[0]
    else:
        return None

# Apply the function to the column
df['image'] = df['images'].apply(string_to_list)

# Create a new column 'First_Value' to store the first value of the nested list in Column1
df['img_webp'] = df['image'].apply(get_first_value)
#
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf_vectorizer.fit_transform(df_filtered['Combined'])
#
# # Calculate cosine similarity
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#
# with open('tfidf_vectorizer.pkl', 'wb') as f:
#     pickle.dump(tfidf_vectorizer, f)
#
# with open('tfidf_matrix.pkl', 'wb') as f:
#     pickle.dump(tfidf_matrix, f)
