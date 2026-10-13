import pandas as pd


def normalize_label(value):
    """Convert common spam/ham labels to a consistent format."""
    if pd.isna(value):
        return None

    label = str(value).strip().lower()

    spam_labels = {"spam", "1", "junk", "unsolicited"}
    ham_labels = {"ham", "0", "not spam", "not_spam", "legitimate"}

    if label in spam_labels:
        return "spam"

    if label in ham_labels:
        return "ham"

    return None


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "text", label_column: str = "label",) -> pd.DataFrame:
    """Validate, normalize, and clean a spam-classification dataset."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Match column names without depending on capitalization or whitespace.
    normalized_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    text_candidates = ["text", "message", "sms", "v2", "email"]
    label_candidates = ["label", "category", "class", "target", "v1"]

    if text_column not in df.columns:
        matched_text = next(
            (
                normalized_columns[name]
                for name in text_candidates
                if name in normalized_columns
            ),
            None,
        )

        if matched_text is None:
            raise ValueError(
                f"Text column not found. Available columns: {list(df.columns)}"
            )

        text_column = matched_text

    if label_column not in df.columns:
        matched_label = next(
            (
                normalized_columns[name]
                for name in label_candidates
                if name in normalized_columns
            ),
            None,
        )

        if matched_label is None:
            raise ValueError(
                f"Label column not found. Available columns: {list(df.columns)}"
            )

        label_column = matched_label

    clean_df = df[[text_column, label_column]].copy()
    clean_df.columns = ["text", "label"]

    clean_df = clean_df.dropna(subset=["text", "label"])

    clean_df["text"] = clean_df["text"].astype(str).str.strip()
    clean_df["label"] = clean_df["label"].apply(normalize_label)

    clean_df = clean_df[(clean_df["text"] != "") & (clean_df["label"].isin(["spam", "ham"]))]
    clean_df = clean_df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if clean_df.empty:
        raise ValueError(
            "No valid records remain after preprocessing. "
            "Check the dataset columns and labels."
        )

    if clean_df["label"].nunique() < 2:
        raise ValueError(
            "Dataset must contain both spam and ham examples."
        )

    return clean_df


def load_and_preprocess_csv(
    csv_path,
    text_column: str = "text",
    label_column: str = "label",
) -> pd.DataFrame:
    """Load a CSV file and preprocess its contents."""

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    return preprocess_dataframe(
        df,
        text_column=text_column,
        label_column=label_column,
    )
