"""
category_vectorizer.py

Purpose:
--------
Creates TF-IDF vectors for the category documents
and saves the trained model.

Implements:

Group Text by Category
↓

Fit TF-IDF Vectorizer
↓

Save Category Vectors

(Figure 2 of the IEEE paper)
"""

import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

from category_builder import build_category_documents


MODEL_DIR = "models"


def train_category_vectorizer():

    print("=" * 60)
    print("BUILDING CATEGORY DOCUMENTS")
    print("=" * 60)

    category_documents = build_category_documents()

    print("\nCreating Category Lists...")

    category_labels = sorted(category_documents.keys())

    documents = [
        category_documents[label]
        for label in category_labels
    ]

    print("\nTraining TF-IDF Vectorizer...")

    vectorizer = TfidfVectorizer(

        ngram_range=(1, 3),

        min_df=2,

        max_df=0.95,

        norm="l2"

    )

    category_vectors = vectorizer.fit_transform(documents)

    print("\nVocabulary Size :", len(vectorizer.vocabulary_))

    print("Category Vector Shape :", category_vectors.shape)

    return vectorizer, category_vectors, category_labels


def save_model(
        vectorizer,
        category_vectors,
        category_labels
):

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\nSaving Model Files...")

    joblib.dump(
        vectorizer,
        f"{MODEL_DIR}/tfidf_vectorizer.pkl"
    )

    joblib.dump(
        category_vectors,
        f"{MODEL_DIR}/category_vectors.pkl"
    )

    joblib.dump(
        category_labels,
        f"{MODEL_DIR}/category_labels.pkl"
    )

    print("Model Saved Successfully.")


def display_category_information(
        labels,
        vectors
):

    print("\n")
    print("=" * 60)
    print("CATEGORY INFORMATION")
    print("=" * 60)

    for i, label in enumerate(labels):

        non_zero = vectors[i].count_nonzero()

        print(f"{label:>3}  ->  {non_zero} TF-IDF Features")


if __name__ == "__main__":

    vectorizer, category_vectors, category_labels = train_category_vectorizer()

    display_category_information(
        category_labels,
        category_vectors
    )

    save_model(
        vectorizer,
        category_vectors,
        category_labels
    )

    print("\n" + "=" * 60)
    print("CATEGORY VECTORIZATION COMPLETED")
    print("=" * 60)