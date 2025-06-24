import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title="Netflix Dashboard", layout="wide")

# 📄 Load data
df = pd.read_csv("netflix_titles.csv")

# 🧠 Recommender System Setup
df_reco = df[['title', 'description']].dropna().reset_index(drop=True)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_reco['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
title_indices = pd.Series(df_reco.index, index=df_reco['title'].str.lower())

def recommend_show(title, top_n=5):
    title = title.lower()
    if title not in title_indices:
        return []
    idx = title_indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_indices = [i[0] for i in similarity_scores]
    return df_reco['title'].iloc[recommended_indices].tolist()

# 🧹 Clean data for visualizations
df.dropna(subset=['date_added', 'country', 'listed_in'], inplace=True)
df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
df['year_added'] = df['date_added'].dt.year

# 🌟 Header Banner
st.markdown("<h1 style='text-align: center; color: red;'>🎬 Netflix Content Explorer & Recommender</h1>", unsafe_allow_html=True)
st.markdown("### Dive into Netflix trends and discover what to watch next 🔥")

st.markdown("---")

# 🔍 Recommender Input
st.subheader("🔍 Recommend Me Something!")
show_input = st.text_input("Enter a Netflix show or movie title (e.g., Narcos, Dark):")
if st.button("Recommend"):
    if show_input:
        results = recommend_show(show_input)
        if results:
            st.success("Top Recommendations 🎯")
            for r in results:
                st.write(f"✅ **{r}**")
        else:
            st.error("❌ Show not found. Try a different title.")

st.markdown("---")

# 📈 Titles Added Over the Years (Trend)
st.subheader("📈 Netflix Content Growth Over the Years")
yearly_counts = df['year_added'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', ax=ax)
ax.set_title("Titles Added to Netflix Each Year", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Titles")
st.pyplot(fig)

st.markdown("---")
# 🎭 Genre Trends Over Time (Multi-Year View)
st.subheader("📊 Genre Trends Over Time")

# Pick top N genres across all years
genre_explode = df.copy()
genre_explode['genre'] = df['listed_in'].str.split(', ')
genre_explode = genre_explode.explode('genre')
top_genres_overall = genre_explode['genre'].value_counts().head(5).index

# Filter only top genres for chart
top_genre_data = genre_explode[genre_explode['genre'].isin(top_genres_overall)]
genre_by_year = top_genre_data.groupby(['year_added', 'genre']).size().unstack().fillna(0)

# Plot it
fig2, ax2 = plt.subplots(figsize=(12, 6))
genre_by_year.plot(kind='bar', stacked=False, ax=ax2)
ax2.set_title("Top 5 Genres Added to Netflix Over Time", fontsize=14)
ax2.set_xlabel("Year")
ax2.set_ylabel("Number of Titles")
plt.xticks(rotation=45)
st.pyplot(fig2)

# 📊 Filters
col1, col2 = st.columns(2)
with col1:
    selected_year = st.selectbox("📅 Select Release Year", sorted(df['year_added'].dropna().unique()))
with col2:
    top_countries = df['country'].dropna().str.split(',').explode().str.strip().value_counts().head(20).index
    selected_country = st.selectbox("🌍 Select Country", top_countries)

# 🎯 Filtered Data
filtered_df = df[(df['year_added'] == selected_year) & (df['country'].str.contains(selected_country))]
# 🎥 Actor & Director Filter
st.subheader("🎥 Explore by Actor or Director")

col1, col2 = st.columns(2)
with col1:
    selected_actor = st.selectbox("Select Actor", 
        df['cast'].dropna().str.split(',').explode().str.strip().value_counts().head(50).index.insert(0, "All"))
with col2:
    selected_director = st.selectbox("Select Director", 
        df['director'].dropna().str.split(',').explode().str.strip().value_counts().head(50).index.insert(0, "All"))

# Apply filter
actor_filter = df['cast'].fillna('').str.contains(selected_actor) if selected_actor != "All" else pd.Series([True] * len(df), index=df.index)
director_filter = df['director'].fillna('').str.contains(selected_director) if selected_director != "All" else pd.Series([True] * len(df), index=df.index)

df_filtered_people = df[actor_filter & director_filter]


# Show results
st.write(f"🎬 Found {len(df_filtered_people)} titles matching filters")
st.dataframe(df_filtered_people[['title', 'type', 'director', 'cast', 'listed_in', 'description']].head(10), use_container_width=True)

# 📈 Content Type Chart
st.subheader(f"📦 Content Types in {selected_country} ({selected_year})")
st.bar_chart(filtered_df['type'].value_counts())

# 🎭 Top Genres
st.subheader("🎭 Top Genres This Year")
genre_list = filtered_df['listed_in'].str.split(', ')
all_genres = [genre for sublist in genre_list.dropna() for genre in sublist]
top_genres = pd.Series(all_genres).value_counts().head(10)
st.bar_chart(top_genres)

# 📃 Sample Titles
st.subheader("📃 Sample Titles")
st.dataframe(filtered_df[['title', 'type', 'listed_in', 'description']].head(10), use_container_width=True)
# ⏱️ Movie Duration Distribution
st.subheader("⏱️ Movie Duration Distribution (in Minutes)")

# Filter movies and extract minutes
movie_df = df[df['type'] == 'Movie'].copy()
movie_df['duration_minutes'] = movie_df['duration'].str.extract(r'(\d+)').astype(float)

# Plot histogram
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.histplot(movie_df['duration_minutes'].dropna(), bins=30, kde=True, ax=ax3, color='crimson')
ax3.set_title("Distribution of Movie Durations")
ax3.set_xlabel("Duration (minutes)")
ax3.set_ylabel("Number of Movies")
st.pyplot(fig3)

# 📺 TV Show Season Count
st.subheader("📺 TV Show Seasons Distribution")

# Filter TV shows and extract seasons
tv_df = df[df['type'] == 'TV Show'].copy()
tv_df['num_seasons'] = tv_df['duration'].str.extract(r'(\d+)').astype(float)

# Plot bar chart
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.countplot(x='num_seasons', data=tv_df, ax=ax4, color='skyblue')
ax4.set_title("Number of Seasons in TV Shows")
ax4.set_xlabel("Seasons")
ax4.set_ylabel("Number of Shows")
st.pyplot(fig4)

# 📝 Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>📊 Project by Jasmin | Final Year CSE | Netflix Data Analysis & Recommender</p>", unsafe_allow_html=True)
