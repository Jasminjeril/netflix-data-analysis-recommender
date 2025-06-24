# 🎬 Netflix Data Analysis & Recommender

Dive deep into Netflix trends, visualize global content growth, and get intelligent show/movie recommendations based on descriptions using NLP!

## 🔍 Features

- 📈 Visualize Netflix content growth over the years
- 🌍 Filter by country and year
- 🎭 Explore top genres and content types
- 🤖 Recommend similar shows using TF-IDF + Cosine Similarity
- 📊 Genre trends, movie duration & season distribution
- 🎥 Actor/Director-based filtering

## 🚀 Tech Stack

- Python
- Streamlit
- Pandas, Seaborn, Matplotlib
- Scikit-learn (for recommender system)

## 🧠 Recommender Logic

> Content-based recommendation using TF-IDF vectorizer over descriptions and cosine similarity to recommend shows similar to the one searched.

## 📸 Screenshots

| Dashboard View | Recommendation |
|----------------|----------------|
| ![Dashboard](assets/dashboard.png) | ![Reco](assets/recommendation.png) |

## 🏁 Run Locally

```bash
pip install -r requirements.txt
streamlit run netflix_dashboard.py
