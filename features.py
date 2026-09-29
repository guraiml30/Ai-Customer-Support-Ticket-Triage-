from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import clean_text


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        sublinear_tf=True,
    )
