"""
centroid_experiment.py

CENTROID REPRESENTATION EXPERIMENT
==================================

Purpose
-------
Compare the current category-document representation with a
category-centroid representation.

CURRENT PAPER-STYLE BASELINE
----------------------------
All training requirements belonging to a category are concatenated
into one large document.

Example:

    Functional:
        requirement 1
        requirement 2
        requirement 3
        ...
             |
             v
        ONE category document
             |
             v
        ONE TF-IDF vector


CENTROID EXPERIMENT
-------------------
Each training requirement is converted into its own TF-IDF vector.

Then all requirement vectors belonging to the same category are
averaged to create a category centroid.

Example:

    Functional:
        Req 1 -> TF-IDF vector
        Req 2 -> TF-IDF vector
        Req 3 -> TF-IDF vector
                  |
                  v
             average vectors
                  |
                  v
          Functional centroid


IMPORTANT
---------
This is an experimental enhancement.

It is NOT claimed to be explicitly described in the IEEE paper.

The purpose is to determine whether the representation of a
category is the main limitation of the current classifier.

Fixed experimental settings:

    Dataset          : PROMISE
    Samples          : 624 after removing PO
    Split            : 80/20 stratified
    Random state     : 42

    TF-IDF:
        max_df       = 0.95
        min_df       = 0.01
        max_features = 2000
        ngram_range  = (1, 2)
        norm         = l2

    Similarity       : Cosine similarity

    General ratio    : 0.80
    L / FT ratio     : 0.75

The dataset and test split remain unchanged so that this experiment
can be compared directly with the current baseline.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing import clean_text
from load_data import load_dataset


# ==========================================================
# CONSTANTS
# ==========================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

GENERAL_RATIO_THRESHOLD = 0.80

UNDERREPRESENTED_RATIO_THRESHOLD = 0.75

UNDERREPRESENTED_CATEGORIES = {
    "L",
    "FT"
}

OUTPUT_FOLDER = "outputs"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==========================================================
# SECTION 1
# LOAD AND PREPROCESS DATASET
# ==========================================================

def prepare_dataset():

    """
    Load PROMISE dataset and apply preprocessing.

    PO category is removed.

    Returns
    -------
    pandas.DataFrame
    """

    print("=" * 70)
    print("LOADING PROMISE DATASET")
    print("=" * 70)

    df = load_dataset()

    # ------------------------------------------------------
    # Remove PO
    # ------------------------------------------------------

    df = df[
        df["class"] != "PO"
    ].reset_index(
        drop=True
    )

    print(
        f"Dataset Size : {len(df)}"
    )

    # ------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------

    print(
        "\nApplying NLP preprocessing..."
    )

    df["Cleaned_Text"] = (
        df["RequirementText"]
        .apply(clean_text)
    )

    print(
        "Preprocessing completed."
    )

    return df


# ==========================================================
# SECTION 2
# CREATE SAME 80/20 SPLIT
# ==========================================================

def create_split(df):

    """
    Create the same stratified 80/20 split used by the
    previous evaluation.

    This is important because it makes the centroid experiment
    directly comparable to the existing 69.60% baseline.
    """

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

    return (
        train_df,
        test_df
    )


# ==========================================================
# SECTION 3
# TRAIN TF-IDF ON INDIVIDUAL TRAINING REQUIREMENTS
# ==========================================================

def train_tfidf(train_df):

    """
    Train TF-IDF on individual training requirements.

    IMPORTANT:

    The vectorizer is fitted ONLY on the training set.

    This prevents test-data leakage.
    """

    print(
        "\nTraining TF-IDF..."
    )

    vectorizer = TfidfVectorizer(

        max_df=0.95,

        min_df=0.01,

        max_features=2000,

        ngram_range=(1, 2),

        norm="l2"

    )

    training_vectors = (
        vectorizer.fit_transform(
            train_df["Cleaned_Text"]
        )
    )

    print(
        f"Vocabulary Size : "
        f"{len(vectorizer.vocabulary_)}"
    )

    print(
        f"Training Vector Shape : "
        f"{training_vectors.shape}"
    )

    return (
        vectorizer,
        training_vectors
    )


# ==========================================================
# SECTION 4
# BUILD CATEGORY CENTROIDS
# ==========================================================

def build_category_centroids(
    train_df,
    training_vectors
):

    """
    Build one centroid vector for each category.

    For every category:

        centroid =
            mean of all training requirement vectors

    The resulting centroid is normalized before cosine
    similarity is calculated.

    Returns
    -------
    category_centroids
        Dictionary:

            category -> centroid vector

    labels
        Sorted category labels
    """

    print(
        "\nBuilding Category Centroids..."
    )

    labels = sorted(
        train_df["class"].unique()
    )

    category_centroids = {}

    # ------------------------------------------------------
    # Process each category
    # ------------------------------------------------------

    for category in labels:

        # Find training rows belonging to category
        category_indices = np.where(
            train_df["class"].values
            == category
        )[0]

        category_vectors = (
            training_vectors[
                category_indices
            ]
        )

        # --------------------------------------------------
        # Calculate mean vector
        # --------------------------------------------------

        centroid = (
            category_vectors.mean(
                axis=0
            )
        )

        # Convert matrix to numpy array
        centroid = np.asarray(
            centroid
        ).reshape(1, -1)

        # --------------------------------------------------
        # Normalize centroid
        # --------------------------------------------------

        norm = np.linalg.norm(
            centroid
        )

        if norm > 0:

            centroid = (
                centroid / norm
            )

        category_centroids[
            category
        ] = centroid

        print(
            f"{category:>3} : "
            f"{len(category_indices):>4} "
            f"training requirements"
        )

    print(
        "\nGenerated "
        f"{len(category_centroids)} "
        "category centroids."
    )

    return (
        category_centroids,
        labels
    )


# ==========================================================
# SECTION 5
# PREDICT REQUIREMENT USING CENTROIDS
# ==========================================================

def predict_requirement(
    requirement,
    vectorizer,
    category_centroids,
    labels
):

    """
    Predict a requirement using category centroids.

    Steps:

        Requirement
             |
             v
        preprocessing
             |
             v
          TF-IDF
             |
             v
        cosine similarity
             |
             v
       category ranking
             |
             v
       maximum similarity
             |
             v
      relative threshold
             |
             v
       final prediction
    """

    # ======================================================
    # PREPROCESS
    # ======================================================

    cleaned = clean_text(
        requirement
    )

    # ======================================================
    # TRANSFORM REQUIREMENT
    # ======================================================

    requirement_vector = (
        vectorizer.transform(
            [cleaned]
        )
    )

    # ======================================================
    # CALCULATE COSINE SIMILARITY
    # ======================================================

    scores = []

    for category in labels:

        centroid = (
            category_centroids[
                category
            ]
        )

        similarity = cosine_similarity(

            requirement_vector,

            centroid

        )[0][0]

        scores.append(
            (
                category,
                float(similarity)
            )
        )

    # ======================================================
    # SORT SCORES
    # ======================================================

    scores.sort(

        key=lambda item: item[1],

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

    thresholds = {}

    for category, score in scores:

        if category in (
            UNDERREPRESENTED_CATEGORIES
        ):

            ratio = (
                UNDERREPRESENTED_RATIO_THRESHOLD
            )

        else:

            ratio = (
                GENERAL_RATIO_THRESHOLD
            )

        threshold = (
            max_similarity * ratio
        )

        thresholds[
            category
        ] = threshold

        if score >= threshold:

            selected_categories.append(
                category
            )

    # ======================================================
    # SINGLE-LABEL PREDICTION
    # ======================================================

    predicted_class = scores[0][0]

    return (

        predicted_class,

        selected_categories,

        scores,

        max_similarity,

        thresholds

    )


# ==========================================================
# SECTION 6
# EVALUATE CENTROID CLASSIFIER
# ==========================================================

def evaluate_centroid():

    """
    Run complete centroid experiment.
    """

    print("\n")
    print("=" * 70)
    print("CATEGORY CENTROID EXPERIMENT")
    print("=" * 70)

    # ======================================================
    # LOAD DATA
    # ======================================================

    df = prepare_dataset()

    # ======================================================
    # SPLIT DATA
    # ======================================================

    train_df, test_df = create_split(
        df
    )

    # ======================================================
    # TRAIN TF-IDF
    # ======================================================

    (
        vectorizer,
        training_vectors

    ) = train_tfidf(
        train_df
    )

    # ======================================================
    # BUILD CENTROIDS
    # ======================================================

    (
        category_centroids,
        labels

    ) = build_category_centroids(

        train_df,

        training_vectors

    )

    # ======================================================
    # PREDICTIONS
    # ======================================================

    y_true = []

    y_pred = []

    prediction_results = []

    print("\n")
    print("=" * 70)
    print("PREDICTING TEST REQUIREMENTS")
    print("=" * 70)

    total = len(test_df)

    for counter, (_, row) in enumerate(
        test_df.iterrows(),
        start=1
    ):

        (
            predicted,
            selected,
            scores,
            max_similarity,
            thresholds

        ) = predict_requirement(

            row["RequirementText"],

            vectorizer,

            category_centroids,

            labels

        )

        # --------------------------------------------------
        # Ground truth
        # --------------------------------------------------

        actual = row["class"]

        y_true.append(
            actual
        )

        y_pred.append(
            predicted
        )

        # --------------------------------------------------
        # Save detailed result
        # --------------------------------------------------

        prediction_results.append({

            "Requirement":
                row["RequirementText"],

            "Actual":
                actual,

            "Predicted":
                predicted,

            "Correct":
                actual == predicted,

            "MaxSimilarity":
                max_similarity,

            "SelectedCategories":
                "|".join(
                    selected
                ),

            "TopCategory":
                scores[0][0],

            "TopSimilarity":
                scores[0][1]

        })

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (
            counter % 100 == 0
            or counter == total
        ):

            print(
                f"Processed "
                f"{counter}/{total}"
            )

    # ======================================================
    # METRICS
    # ======================================================

    accuracy = accuracy_score(

        y_true,

        y_pred

    )

    weighted_precision = (
        precision_score(

            y_true,

            y_pred,

            average="weighted",

            zero_division=0

        )
    )

    weighted_recall = (
        recall_score(

            y_true,

            y_pred,

            average="weighted",

            zero_division=0

        )
    )

    weighted_f1 = (
        f1_score(

            y_true,

            y_pred,

            average="weighted",

            zero_division=0

        )
    )

    macro_precision = (
        precision_score(

            y_true,

            y_pred,

            average="macro",

            zero_division=0

        )
    )

    macro_recall = (
        recall_score(

            y_true,

            y_pred,

            average="macro",

            zero_division=0

        )
    )

    macro_f1 = (
        f1_score(

            y_true,

            y_pred,

            average="macro",

            zero_division=0

        )
    )

    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    report = classification_report(

        y_true,

        y_pred,

        labels=labels,

        zero_division=0

    )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=labels

    )

    cm_df = pd.DataFrame(

        cm,

        index=labels,

        columns=labels

    )

    # ======================================================
    # DISPLAY RESULTS
    # ======================================================

    print("\n")
    print("=" * 70)
    print("CENTROID MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Weighted Precision : "
        f"{weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{weighted_recall:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{macro_f1:.4f}"
    )

    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    print("\n")
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        report
    )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    print("\n")
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        cm_df
    )

    # ======================================================
    # SAVE PREDICTIONS
    # ======================================================

    prediction_df = pd.DataFrame(
        prediction_results
    )

    prediction_path = os.path.join(

        OUTPUT_FOLDER,

        "centroid_predictions.csv"

    )

    prediction_df.to_csv(

        prediction_path,

        index=False

    )

    # ======================================================
    # SAVE CONFUSION MATRIX
    # ======================================================

    cm_path = os.path.join(

        OUTPUT_FOLDER,

        "centroid_confusion_matrix.csv"

    )

    cm_df.to_csv(
        cm_path
    )

    # ======================================================
    # SAVE METRICS
    # ======================================================

    metrics_path = os.path.join(

        OUTPUT_FOLDER,

        "centroid_metrics.txt"

    )

    with open(

        metrics_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            "CATEGORY CENTROID EXPERIMENT\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "Dataset: PROMISE\n"
        )

        file.write(
            f"Dataset Size: {len(df)}\n"
        )

        file.write(
            f"Training Samples: "
            f"{len(train_df)}\n"
        )

        file.write(
            f"Testing Samples: "
            f"{len(test_df)}\n\n"
        )

        file.write(
            "TF-IDF Configuration\n"
        )

        file.write(
            "max_df=0.95\n"
        )

        file.write(
            "min_df=0.01\n"
        )

        file.write(
            "max_features=2000\n"
        )

        file.write(
            "ngram_range=(1,2)\n"
        )

        file.write(
            "norm=l2\n\n"
        )

        file.write(
            "Threshold Configuration\n"
        )

        file.write(
            "General ratio=0.80\n"
        )

        file.write(
            "L/FT ratio=0.75\n\n"
        )

        file.write(
            "Representation\n"
        )

        file.write(
            "Mean TF-IDF vector per category\n\n"
        )

        file.write(
            f"Accuracy={accuracy:.6f}\n"
        )

        file.write(
            f"Weighted Precision="
            f"{weighted_precision:.6f}\n"
        )

        file.write(
            f"Weighted Recall="
            f"{weighted_recall:.6f}\n"
        )

        file.write(
            f"Weighted F1="
            f"{weighted_f1:.6f}\n"
        )

        file.write(
            f"Macro Precision="
            f"{macro_precision:.6f}\n"
        )

        file.write(
            f"Macro Recall="
            f"{macro_recall:.6f}\n"
        )

        file.write(
            f"Macro F1="
            f"{macro_f1:.6f}\n"
        )

    # ======================================================
    # FINAL MESSAGE
    # ======================================================

    print("\n")
    print("=" * 70)
    print("CENTROID EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        "  outputs/centroid_predictions.csv"
    )

    print(
        "  outputs/centroid_confusion_matrix.csv"
    )

    print(
        "  outputs/centroid_metrics.txt"
    )

    return {

        "accuracy":
            accuracy,

        "weighted_precision":
            weighted_precision,

        "weighted_recall":
            weighted_recall,

        "weighted_f1":
            weighted_f1,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1

    }


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    evaluate_centroid()