"""
Step 1: clean the raw TMDB 5000 dataset and build a compact CSV for the app.

Put these two Kaggle files in data/raw/ :
    tmdb_5000_movies.csv
    tmdb_5000_credits.csv
Then run:  python prepare_data.py
"""
import ast
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/movies_processed.csv")


def names(json_str, limit=None):
    """Parse a JSON-like string column and return a list of 'name' values."""
    try:
        items = ast.literal_eval(json_str)
    except (ValueError, SyntaxError):
        return []
    result = [i["name"] for i in items]
    return result[:limit] if limit else result


def director(json_str):
    try:
        for member in ast.literal_eval(json_str):
            if member.get("job") == "Director":
                return member["name"]
    except (ValueError, SyntaxError):
        pass
    return ""


def squash(words):
    """'Sam Worthington' -> 'samworthington' so each person/genre is ONE token."""
    return [w.replace(" ", "").lower() for w in words]


def main():
    movies = pd.read_csv(RAW / "tmdb_5000_movies.csv")
    credits = pd.read_csv(RAW / "tmdb_5000_credits.csv")

    df = movies.merge(
        credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id"
    )

    df = df[
        ["id", "title", "overview", "genres", "keywords", "cast", "crew",
         "vote_average", "vote_count", "release_date"]
    ].dropna(subset=["overview"]).drop_duplicates(subset="title").reset_index(drop=True)

    df["genres_list"] = df["genres"].apply(names)
    df["keywords_list"] = df["keywords"].apply(names)
    df["cast_list"] = df["cast"].apply(lambda x: names(x, limit=3))
    df["director"] = df["crew"].apply(director)

    # The "tags" column is the text that TF-IDF will vectorize
    df["tags"] = (
        df["overview"].str.lower()
        + " " + df["genres_list"].apply(squash).str.join(" ")
        + " " + df["keywords_list"].apply(squash).str.join(" ")
        + " " + df["cast_list"].apply(squash).str.join(" ")
        + " " + df["director"].apply(lambda d: squash([d])[0] if d else "")
    )

    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["genres"] = df["genres_list"].str.join(", ")
    df["cast"] = df["cast_list"].str.join(", ")

    out = df[["id", "title", "overview", "genres", "cast", "director",
              "vote_average", "vote_count", "release_year", "tags"]]
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} movies to {OUT}")


if __name__ == "__main__":
    main()
