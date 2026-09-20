
"""
inspect_category_vectors.py

Purpose:
--------
Inspect the top TF-IDF keywords for each category vector
and compare them with Table 3 of the IEEE paper.

Author: NLPReqClassifier
"""

import os
import joblib
import pandas as pd

# ----------------------------------------------------------
# Load Saved Model
# ----------------------------------------------------------

MODEL_DIR = "models"

vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
category_vectors = joblib.load(os.path.join(MODEL_DIR, "category_vectors.pkl"))
category_labels = joblib.load(os.path.join(MODEL_DIR, "category_labels.pkl"))

# ----------------------------------------------------------
# Get Vocabulary
# ----------------------------------------------------------

feature_names = vectorizer.get_feature_names_out()

# ----------------------------------------------------------
# Create Output Folder
# ----------------------------------------------------------

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []

print("=" * 80)
print("TOP TF-IDF KEYWORDS FOR EACH CATEGORY")
print("=" * 80)

# ----------------------------------------------------------
# Display Top Keywords
# ----------------------------------------------------------

TOP_N = 20

for idx, category in enumerate(category_labels):

    print("\n" + "=" * 60)
    print(f"CATEGORY : {category}")
    print("=" * 60)

    # Convert sparse vector to dense array
    vector = category_vectors[idx].toarray().flatten()

    # Sort by TF-IDF score
    top_indices = vector.argsort()[::-1][:TOP_N]

    for rank, feature_index in enumerate(top_indices, start=1):

        keyword = feature_names[feature_index]
        score = vector[feature_index]

        print(f"{rank:2}. {keyword:<35} {score:.4f}")

        results.append({
            "Category": category,
            "Rank": rank,
            "Keyword": keyword,
            "TF-IDF Score": round(float(score), 4)
        })

# ----------------------------------------------------------
# Save Results
# ----------------------------------------------------------

df = pd.DataFrame(results)

csv_path = os.path.join(
    OUTPUT_DIR,
    "top_category_keywords.csv"
)

df.to_csv(csv_path, index=False)

print("\n" + "=" * 80)
print("Keyword inspection completed.")
print(f"CSV saved to: {csv_path}")
print("=" * 80)