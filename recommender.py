import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("netflix_titles.csv")
df = df[['title', 'description']].dropna().reset_index(drop=True)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Title-to-index mapping
title_indices = pd.Series(df.index, index=df['title'].str.lower())

# Recommendation function
def recommend_show(title, top_n=5):
    title = title.lower()
    if title not in title_indices:
        return f"'{title}' not found in the dataset."

    idx = title_indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommended_indices = [i[0] for i in similarity_scores]
    return df['title'].iloc[recommended_indices].tolist()

# Try it!
if __name__ == "__main__":
    show = input("Enter a show/movie title: ")
    results = recommend_show(show)
    print("\nRecommended Shows:")
    for r in results:
        print("✅", r)
