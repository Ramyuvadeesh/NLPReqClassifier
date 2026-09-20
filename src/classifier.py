from load_data import load_dataset
from preprocessing import clean_text
from results import generate_all_outputs

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import pandas as pd


# ==========================================================
# STEP 1 : LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = load_dataset()

# Remove PO class (only one instance)
df = df[df["class"] != "PO"].reset_index(drop=True)

# ==========================================================
# STEP 2 : PREPROCESSING
# ==========================================================

print("\nPreprocessing Requirements...")

df["Cleaned_Text"] = df["RequirementText"].apply(clean_text)

# ==========================================================
# STEP 3 : TRAIN TEST SPLIT
# ==========================================================

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    df["Cleaned_Text"],
    df["class"],
    test_size=0.20,
    random_state=42,
    stratify=df["class"]
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ==========================================================
# STEP 4 : TF-IDF
# ==========================================================

print("\nCreating TF-IDF Vectors...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    norm="l2"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Vocabulary Size : {len(vectorizer.vocabulary_)}")

# ==========================================================
# STEP 5 : COSINE SIMILARITY
# ==========================================================

print("\nCalculating Cosine Similarity...")

similarity_matrix = cosine_similarity(
    X_test_tfidf,
    X_train_tfidf
)

print(f"Similarity Matrix Shape : {similarity_matrix.shape}")

# ==========================================================
# STEP 6 : PREDICTION
# ==========================================================

nearest_indices = similarity_matrix.argmax(axis=1)

y_train = y_train.reset_index(drop=True)
y_pred = y_train.iloc[nearest_indices].values

# ==========================================================
# STEP 7 : EVALUATION
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ==========================================================
# STEP 8 : CLASSIFICATION REPORT
# ==========================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# ==========================================================
# STEP 9 : CONFUSION MATRIX
# ==========================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

labels = sorted(df["class"].unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)

# ==========================================================
# STEP 10 : SAMPLE PREDICTIONS
# ==========================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

for i in range(min(10, len(X_test))):

    print(f"\nRequirement {i+1}")
    print("-" * 60)

    print("Requirement:")
    print(X_test.iloc[i])

    print("\nActual Class    :", y_test.iloc[i])
    print("Predicted Class :", y_pred[i])

# ==========================================================
# STEP 11 : SAVE OUTPUT FILES
# ==========================================================

generate_all_outputs(
    df=df,
    X_test=X_test,
    y_test=y_test,
    y_pred=y_pred,
    labels=labels,
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1=f1
)

print("\n" + "=" * 60)
print("ALL OUTPUT FILES GENERATED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")

print("outputs/metrics.txt")
print("outputs/predictions.csv")
print("outputs/classification_report.csv")
print("outputs/confusion_matrix.csv")
print("outputs/confusion_matrix.png")
print("outputs/class_distribution.png")

print("\nProject Execution Completed Successfully.")