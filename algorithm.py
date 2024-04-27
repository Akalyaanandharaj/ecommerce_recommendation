import pandas as pd
from PIL import Image
from io import BytesIO
import requests, ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random, pickle

# # Sample perfume data with occasion and age group tag
#
df = pd.read_csv("data/ecommerce_data.csv")
df['description_cleaned'] = df['description'].replace(r'[^a-zA-Z0-9\s]', '', regex=True)
df['name_cleaned'] = df['name'].replace(r'[^a-zA-Z0-9\s]', '', regex=True)

def string_to_list(string):
    return ast.literal_eval(string)
df['image'] = df['images'].apply(string_to_list)

def get_first_value(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return lst[0]
    else:
        return None
df['img_webp'] = df['image'].apply(get_first_value)

# Combine description, occasion, and age group into a single text for TF-IDF Vectorizer
df['Combined'] = df['description_cleaned'] + df['name_cleaned']
#
# # Initialize TF-IDF Vectorizer
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#
# # Fit and transform the combined text
# tfidf_matrix = tfidf_vectorizer.fit_transform(df['Combined'])
#
# # Calculate cosine similarity
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# with open('tfidf_vectorizer.pkl', 'wb') as f:
#     pickle.dump(tfidf_vectorizer, f)
#
# with open('cosine_sim.pkl', 'wb') as f:
#     pickle.dump(cosine_sim, f)
# # # # Function to get user input
# def get_user_input():
#     occasion = input("Enter the occasion (formal/beachwear/casual/workout): ").strip().capitalize()
#     material = input("Enter the Material (cotton/silk/wool) : ").strip().capitalize()
#     style = input("Enter the Style (boho/vintage/modern) : ").strip().lower()
#     color = input("Enter the color  : ").strip().lower()
#     return occasion, material, style, color
# # #
def get_recommendations(occasion, age_group, description, cosine_sim, df,tfidf_vectorizer, top_n=3):
    # Create a combined query
    query = description + ' ' + occasion + ' ' + age_group

    # Transform the query using the trained TF-IDF Vectorizer
    query_vector = tfidf_vectorizer.transform([query])

    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Combined'])

    # Calculate cosine similarity between query vector and perfume vectors
    sim_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Get indices of top N most similar perfumes
    sim_scores = sim_scores[0]
    sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    perfume_indices = [i[0] for i in sim_scores]

    results = df[['name', 'img_webp']].iloc[perfume_indices].reset_index(drop=True)
    return results.to_json()
# #
# # # # Load TF-IDF Vectorizer
# # with open('tfidf_vectorizer.pkl', 'rb') as f:
# #     tfidf_vectorizer = pickle.load(f)
# #
# # # Load Cosine Similarity Matrix
# # with open('cosine_sim.pkl', 'rb') as f:
# #     cosine_sim = pickle.load(f)
# #
# # # Get user input
# # occasion, age_group, description = get_user_input()
# #
# # print (occasion, age_group, description)
# #
# # # Get perfume recommendations
# # recommendations = get_recommendations(occasion, age_group, description, cosine_sim, df)
# #
# # print(recommendations)
# #
# #
# #
