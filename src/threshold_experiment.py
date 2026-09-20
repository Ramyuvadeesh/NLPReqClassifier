"""
threshold_experiment.py

Controlled Threshold Ablation Experiment
=========================================

Purpose
-------
Determine how the relative cosine-similarity threshold affects
classification performance.

Only the threshold is changed.

Everything else remains fixed:

    Dataset              : PROMISE
    Dataset size         : 624 (PO removed)
    Split                : 80/20 stratified
    Random state         : 42
    Preprocessing        : existing preprocessing.py
    Category documents   : training data only
    TF-IDF               :
        max_df=0.95
        min_df=0.01
        max_features=2000
        ngram_range=(1,2)
        norm='l2'

Thresholds tested:
    0.60
    0.65
    0.70
    0.75
    0.80
    0.85
    0.90

IMPORTANT
---------
This is an experimental ablation.

The paper explicitly describes a general ratio threshold of 0.80
and a temporary 0.75 threshold for underrepresented categories
such as L and FT.

The other threshold values are NOT claimed to come from the paper.
They are tested only to understand threshold sensitivity.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os

import numpy as np
import pandas as pd

from collections import defaultdict

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import clean_text
from load_data import load_dataset


# ==========================================================
# CONSTANTS
# ==========================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

OUTPUT_FOLDER = "outputs"

THRESHOLDS = [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

UNDERREPRESENTED_CATEGORIES = {
    "L",
    "FT"
}

UNDERREPRESENTED_THRESHOLD = 0.75


# ==========================================================
# DATASET PREPARATION
# ==========================================================

def prepare_dataset():

    print("=" * 70)
    print("LOADING PROMISE DATASET")
    print("=" * 70)

    df = load_dataset()

    # Remove PO
    df = df[
        df["class"] != "PO"
    ].reset_index(drop=True)

    print(
        f"Dataset Size : {len(df)}"
    )

    print(
        "\nApplying NLP preprocessing..."
    )

    df["Cleaned_Text"] = (
        df["RequirementText"]
        .apply(clean_text)
    )

    return df


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

def create_split(df):

    print(
        "\nCreating 80/20 stratified split..."
    )

    train_df, test_df = train_test_split(

        df,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=df["class"]

    )

    train_df = train_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    print(
        f"Training Samples : {len(train_df)}"
    )

    print(
        f"Testing Samples  : {len(test_df)}"
    )

    return train_df, test_df


# ==========================================================
# BUILD CATEGORY DOCUMENTS
# ==========================================================

def build_category_documents(train_df):

    print(
        "\nBuilding category documents..."
    )

    category_documents = defaultdict(list)

    for _, row in train_df.iterrows():

        category = row["class"]

        text = row["Cleaned_Text"]

        category_documents[
            category
        ].append(text)

    # Concatenate requirements within each category
    for category in category_documents:

        category_documents[
            category
        ] = " ".join(
            category_documents[category]
        )

    print(
        f"Generated "
        f"{len(category_documents)} "
        f"category documents."
    )

    return category_documents


# ==========================================================
# TRAIN TF-IDF
# ==========================================================

def train_tfidf(category_documents):

    print(
        "\nTraining TF-IDF..."
    )

    labels = sorted(
        category_documents.keys()
    )

    documents = [
        category_documents[label]
        for label in labels
    ]

    vectorizer = TfidfVectorizer(

        max_df=0.95,

        min_df=0.01,

        max_features=2000,

        ngram_range=(1, 2),

        norm="l2"

    )

    category_vectors = (
        vectorizer.fit_transform(
            documents
        )
    )

    print(
        f"Vocabulary Size : "
        f"{len(vectorizer.vocabulary_)}"
    )

    print(
        f"Category Vector Shape : "
        f"{category_vectors.shape}"
    )

    return (
        vectorizer,
        category_vectors,
        labels
    )


# ==========================================================
# PREDICT ONE REQUIREMENT
# ==========================================================

def predict_requirement(
    requirement,
    vectorizer,
    category_vectors,
    labels,
    ratio_threshold
):

    cleaned = clean_text(
        requirement
    )

    # Transform requirement using
    # training vocabulary
    requirement_vector = (
        vectorizer.transform(
            [cleaned]
        )
    )

    # ======================================================
    # COSINE SIMILARITY
    # ======================================================

    similarities = cosine_similarity(

        requirement_vector,

        category_vectors

    )[0]

    # ======================================================
    # RANK CATEGORIES
    # ======================================================

    scores = []

    for index, category in enumerate(
        labels
    ):

        scores.append(
            (
                category,
                float(
                    similarities[index]
                )
            )
        )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # ======================================================
    # MAXIMUM SIMILARITY
    # ======================================================

    max_similarity = scores[0][1]

    # ======================================================
    # THRESHOLD FILTERING
    # ======================================================

    selected_categories = []

    for category, score in scores:

        # L and FT retain the paper's
        # temporary 0.75 threshold.
        #
        # For the threshold experiment,
        # the tested threshold is applied
        # to all other categories.

        if category in UNDERREPRESENTED_CATEGORIES:

            effective_ratio = (
                UNDERREPRESENTED_THRESHOLD
            )

        else:

            effective_ratio = (
                ratio_threshold
            )

        threshold = (
            max_similarity
            * effective_ratio
        )

        if score >= threshold:

            selected_categories.append(
                category
            )

    # ======================================================
    # SINGLE LABEL
    # ======================================================

    # PROMISE has one ground-truth label.
    #
    # For conventional accuracy/precision/recall/F1,
    # the highest-similarity category is used.

    predicted_class = scores[0][0]

    return (
        predicted_class,
        selected_categories,
        scores,
        max_similarity
    )


# ==========================================================
# RUN ONE THRESHOLD
# ==========================================================

def evaluate_threshold(
    threshold,
    test_df,
    vectorizer,
    category_vectors,
    labels
):

    y_true = []

    y_pred = []

    selected_count = 0

    for _, row in test_df.iterrows():

        (
            predicted,
            selected,
            scores,
            max_similarity

        ) = predict_requirement(

            row["RequirementText"],

            vectorizer,

            category_vectors,

            labels,

            threshold

        )

        y_true.append(
            row["class"]
        )

        y_pred.append(
            predicted
        )

        if selected:

            selected_count += 1

    # ======================================================
    # METRICS
    # ======================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    return {

        "Threshold":
            threshold,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "Selected_Count":
            selected_count,

        "Total_Test":
            len(test_df)

    }


# ==========================================================
# MAIN EXPERIMENT
# ==========================================================

def run_experiment():

    print("\n")
    print("=" * 70)
    print("CONTROLLED THRESHOLD ABLATION EXPERIMENT")
    print("=" * 70)

    # ======================================================
    # LOAD DATA
    # ======================================================

    df = prepare_dataset()

    # ======================================================
    # SPLIT
    # ======================================================

    train_df, test_df = create_split(
        df
    )

    # ======================================================
    # CATEGORY DOCUMENTS
    # ======================================================

    category_documents = (
        build_category_documents(
            train_df
        )
    )

    # ======================================================
    # TF-IDF
    # ======================================================

    (
        vectorizer,
        category_vectors,
        labels

    ) = train_tfidf(
        category_documents
    )

    # ======================================================
    # TEST THRESHOLDS
    # ======================================================

    results = []

    print("\n")
    print("=" * 70)
    print("TESTING THRESHOLDS")
    print("=" * 70)

    for threshold in THRESHOLDS:

        print(
            f"\nTesting threshold = "
            f"{threshold:.2f}"
        )

        result = evaluate_threshold(

            threshold,

            test_df,

            vectorizer,

            category_vectors,

            labels

        )

        results.append(
            result
        )

        print(
            f"Accuracy  : "
            f"{result['Accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{result['Precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{result['Recall']:.4f}"
        )

        print(
            f"F1        : "
            f"{result['F1']:.4f}"
        )

    # ======================================================
    # CREATE RESULTS DATAFRAME
    # ======================================================

    results_df = pd.DataFrame(
        results
    )

    # ======================================================
    # DISPLAY FINAL TABLE
    # ======================================================

    print("\n")
    print("=" * 70)
    print("THRESHOLD EXPERIMENT RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "Threshold",
                "Accuracy",
                "Precision",
                "Recall",
                "F1"
            ]
        ].to_string(
            index=False
        )
    )

    # ======================================================
    # BEST THRESHOLD
    # ======================================================

    best_index = (
        results_df["F1"].idxmax()
    )

    best_row = (
        results_df.loc[
            best_index
        ]
    )

    print("\n")
    print("=" * 70)
    print("BEST THRESHOLD BY WEIGHTED F1")
    print("=" * 70)

    print(
        f"Threshold : "
        f"{best_row['Threshold']:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{best_row['Accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{best_row['Precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_row['Recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_row['F1']:.4f}"
    )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    output_path = os.path.join(

        OUTPUT_FOLDER,

        "threshold_experiment.csv"

    )

    results_df.to_csv(

        output_path,

        index=False

    )

    print("\n")
    print(
        f"Results saved to: "
        f"{output_path}"
    )

    print("\n")
    print("=" * 70)
    print("THRESHOLD EXPERIMENT COMPLETED")
    print("=" * 70)

    return results_df


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    run_experiment()