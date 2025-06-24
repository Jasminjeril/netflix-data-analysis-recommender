# 🎬 Netflix Data Analysis & Recommender
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
Dive deep into Netflix trends, visualize global content growth, and get intelligent show/movie recommendations based on descriptions using NLP!

---

## 🔍 Features

- 📈 Visualize Netflix content growth over the years  
- 🌍 Filter by country and year  
- 🎭 Explore top genres and content types  
- 🤖 Recommend similar shows using TF-IDF + Cosine Similarity  
- 📊 Genre trends, movie duration & season distribution  
- 🎥 Actor/Director-based filtering  

---

## 🚀 Tech Stack

- Python  
- Streamlit  
- Pandas, Seaborn, Matplotlib  
- Scikit-learn (for recommender system)  

---

## 🧠 Recommender Logic

Content-based recommendation using TF-IDF vectorizer over descriptions and cosine similarity to recommend shows similar to the one searched.

---

## 📸 Screenshots

### 📊 Dashboard View  
![Dashboard Screenshot](assets/dashboard%20images/Screenshot%202025-06-24%20120140.png)

### 🤖 Recommendation  
![Recommendation Screenshot](assets/recommendation%20images/Screenshot%202025-06-24%20115952.png)

---

## 🏁 Run Locally

```bash
pip install -r requirements.txt
streamlit run netflix_dashboard.py
