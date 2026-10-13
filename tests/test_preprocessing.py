import pandas as pd
import pytest

from app.pipeline.preprocessing import preprocess_dataframe


def test_preprocessing_normalizes_labels():
    df = pd.DataFrame({
        "text": ["Win a prize", "Hello friend", "   "],
        "label": ["SPAM", "ham", "spam"],
    })

    result = preprocess_dataframe(df)

    assert list(result.columns) == ["text", "label"]
    assert set(result["label"]) == {"spam", "ham"}
    assert not (result["text"] == "").any()


def test_preprocessing_accepts_common_column_names():
    df = pd.DataFrame({
        "v2": ["Claim your prize", "See you soon"],
        "v1": ["spam", "ham"],
    })

    result = preprocess_dataframe(df)

    assert len(result) == 2
    assert set(result["label"]) == {"spam", "ham"}


def test_preprocessing_rejects_empty_dataset():
    with pytest.raises(ValueError):
        preprocess_dataframe(pd.DataFrame(columns=["text", "label"]))


def test_preprocessing_rejects_one_class():
    df = pd.DataFrame({
        "text": ["first spam", "another spam"],
        "label": ["spam", "spam"],
    })

    with pytest.raises(ValueError):
        preprocess_dataframe(df)
