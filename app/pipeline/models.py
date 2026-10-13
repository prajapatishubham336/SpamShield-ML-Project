from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB

from app.pipeline.features import build_vectorizer


def build_models() -> dict:
    """Return fresh, untrained model pipelines."""
    return {
        "logistic_regression": Pipeline([
            ("tfidf", build_vectorizer()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "multinomial_nb": Pipeline([
            ("tfidf", build_vectorizer()),
            ("classifier", MultinomialNB(alpha=0.5)),
        ]),
        "complement_nb": Pipeline([
            ("tfidf", build_vectorizer()),
            ("classifier", ComplementNB(alpha=0.5)),
        ]),
    }