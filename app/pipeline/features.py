from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 20000,
    ngram_range: tuple = (1, 2),
    min_df: int = 1,
) -> TfidfVectorizer:
    """Create a CPU-friendly TF-IDF vectorizer."""
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        dtype=float,
    )