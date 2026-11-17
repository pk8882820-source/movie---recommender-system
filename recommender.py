"""Content-based recommender: TF-IDF + cosine similarity."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, csv_path="data/movies_processed.csv"):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=10000, ngram_range=(1, 2)
        )
        self.matrix = self.vectorizer.fit_transform(self.df["tags"].fillna(""))
        self._index = pd.Series(self.df.index, index=self.df["title"])

    @property
    def titles(self):
        return sorted(self.df["title"].tolist())

    def recommend(self, title, n=5, min_votes=0):
        """Return the n movies most similar to `title`."""
        idx = self._index[title]
        # similarity of ONE movie vs all movies (avoids a huge NxN matrix)
        scores = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        scores[idx] = -1  # exclude the movie itself

        order = np.argsort(scores)[::-1]
        order = [i for i in order if self.df.at[i, "vote_count"] >= min_votes][:n]

        result = self.df.iloc[order].copy()
        result["similarity"] = scores[order]
        return result.drop(columns=["tags"])

    def details(self, title):
        return self.df.loc[self._index[title]]
