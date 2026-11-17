import os
import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Building the model...")
def load_model():
    return MovieRecommender("data/movies_processed.csv")


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    try:
        key = st.secrets["TMDB_API_KEY"]
    except Exception:
        key = os.getenv("TMDB_API_KEY")

    if not key:
        return None

    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}",
            params={"api_key": key},
            timeout=5
        )
        r.raise_for_status()
        path = r.json().get("poster_path")

        if path:
            return f"https://image.tmdb.org/t/p/w500{path}"
        return None

    except Exception:
        return None




model = load_model()

st.title("🎬 Movie Recommendation System")
st.caption("Content-based filtering using TF-IDF and cosine similarity on the TMDB 5000 dataset.")

with st.sidebar:
    st.header("Settings")
    n = st.slider("Number of recommendations", 3, 12, 6)
    min_votes = st.slider("Minimum vote count", 0, 2000, 100, step=50)
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Overview, genres, keywords, cast & director are merged into text.\n"
        "2. TF-IDF turns each movie into a vector.\n"
        "3. Cosine similarity finds the closest movies."
    )

selected = st.selectbox("Pick a movie you like:", model.titles, index=None,
                        placeholder="Start typing a title...")

if st.button("Recommend", type="primary", disabled=selected is None):
    info = model.details(selected)
    with st.expander(f"About: {selected}", expanded=False):
        st.write(info["overview"])
        st.write(f"**Genres:** {info['genres']}  |  **Director:** {info['director']}")

    recs = model.recommend(selected, n=n, min_votes=min_votes)
    st.subheader("You might also like")

    cols_per_row = 3
    for start in range(0, len(recs), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (_, row) in zip(cols, recs.iloc[start:start + cols_per_row].iterrows()):
            with col:
                poster = fetch_poster(row["id"])
                if poster:
                    st.image(poster, use_container_width=True)
                year = "" if row["release_year"] != row["release_year"] else f" ({int(row['release_year'])})"
                st.markdown(f"#### {row['title']}{year}")
                st.progress(float(max(0, min(1, row["similarity"]))),
                            text=f"Match: {row['similarity']:.0%}")
                st.write(f"⭐ {row['vote_average']:.1f}  ·  {row['genres']}")
                st.caption(row["overview"][:200] + "...")
