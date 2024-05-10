import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def get_recommendations(occasion, details, style, color, df,tfidf_vectorizer, tfidf_matrix, top_n=3):
    # Create a combined query
    query = ' '.join([occasion, details, style, color])

    # Transform the query using the trained TF-IDF Vectorizer
    query_vector = tfidf_vectorizer.transform([query])

    # Calculate cosine similarity between query vector and perfume vectors
    sim_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Get indices of top N most similar perfumes
    sim_scores = sim_scores[0]
    sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    product_indices = [i[0] for i in sim_scores]

    results = df[['name', 'img_webp']].iloc[product_indices].reset_index(drop=True)
    return results.to_json()
