import joblib
import pandas as pd
from app.config import MODEL_PATH, RAW_DATA_DIR
from sklearn.metrics import classification_report, confusion_matrix

model = joblib.load(MODEL_PATH)
df = pd.read_csv(RAW_DATA_DIR / "spam.csv")
sample = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

X = sample["text"]

print("Unique messages:", len(X))
print("Pipeline prediction:", model.predict(X))
print("Pipeline output count:", len(model.predict(X)))
sample["prediction"] = model.predict(X)

# Test each pipeline stage separately
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

features = vectorizer.transform(X)

print("\nTF-IDF shape:", features.shape)
print("Classifier prediction shape:", classifier.predict(features).shape)
print("Classifier predictions:", classifier.predict(features))
print("Classifier probability shape:", classifier.predict_proba(features).shape)

print("\n=== Single-message test ===")
for _, row in sample.head(5).iterrows():
    result = model.predict([row["text"]])
    print("Actual:", row["label"], "| Predicted:", result)





print("\n=== MODEL EVALUATION ===")
print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(sample["label"], sample["prediction"], labels=["ham", "spam"]))

print("\nClassification report:")
print(classification_report(
    sample["label"],
    sample["prediction"],
    labels=["ham", "spam"],
    zero_division=0
))
