# 🎬 Movie Recommendation System

A content-based movie recommender built on the **TMDB 5000 dataset**.
Pick a movie and get the most similar ones, ranked by cosine similarity.

**Tech stack:** Python · pandas · NumPy · scikit-learn · TF-IDF Vectorizer · Cosine Similarity · Streamlit · Git/GitHub · Streamlit Community Cloud

## How it works
1. `prepare_data.py` merges movies + credits, parses JSON columns, and builds a `tags` text field
   (overview + genres + keywords + top 3 cast + director).
2. `TfidfVectorizer` converts every movie's tags into a weighted vector.
3. `cosine_similarity` compares the chosen movie against all others; top-N are returned.
4. `app.py` serves it through a Streamlit UI.

## Project structure
```
movie-recommender/
├── app.py                 # Streamlit UI
├── recommender.py         # TF-IDF + cosine similarity engine
├── prepare_data.py        # data cleaning / feature engineering
├── requirements.txt
├── data/
│   ├── raw/               # (git-ignored) Kaggle CSVs
│   └── movies_processed.csv
└── .streamlit/secrets.toml.example
```

## Run locally
```bash
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# download from Kaggle: "TMDB 5000 Movie Dataset" -> put both CSVs in data/raw/
python prepare_data.py
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push the repo to GitHub (commit `data/movies_processed.csv`).
2. Go to https://share.streamlit.io → **New app** → select repo, branch `main`, file `app.py`.
3. (Optional) add `TMDB_API_KEY` under **Settings → Secrets** to show posters.

## Ideas for improvement
- Stemming (NLTK) · genre filter · hybrid with ratings · search by actor/director
