"""
evaluation.py

Research Evaluation Module
==========================

Evaluates the threshold-based cosine similarity classifier
using the PROMISE NFR dataset.

Version 1
---------
80/20 Stratified Hold-Out Evaluation

Version 2
---------
Stratified 5-Fold Cross Validation

Methodology implemented:
    1. Load PROMISE dataset
    2. Remove PO category
    3. Preprocess requirements
    4. Split into training/testing data
    5. Build one category document per category
    6. Train TF-IDF on TRAINING CATEGORY DOCUMENTS ONLY
    7. Create category TF-IDF vectors
    8. Transform each test requirement
    9. Calculate cosine similarity
   10. Find maximum similarity
   11. Apply relative ratio threshold
   12. Predict the highest-ranked category
   13. Calculate evaluation metrics

IMPORTANT
---------
The paper's Figure 2 uses a similarity-ratio filtering mechanism:

    similarity >= max_similarity * ratio_threshold

General ratio threshold:
    0.80

Underrepresented categories:
    L  -> 0.75
    FT -> 0.75

The paper's author confirmed the 0.80 general threshold and
temporary 0.75 threshold for underrepresented categories.

For conventional single-label evaluation, the highest-similarity
category is used as the predicted class.

This single-label conversion is an implementation assumption
because the paper does not provide complete evaluation code
for converting potentially multiple selected categories into
one label.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os

import numpy as np
import pandas as pd

from collections import defaultdict

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

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

N_SPLITS = 5

# General ratio threshold described by the paper
GENERAL_RATIO_THRESHOLD = 0.80

# Temporary threshold for underrepresented categories
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
# DATASET PREPARATION
# ==========================================================

def prepare_dataset():

    """
    Load and preprocess the PROMISE dataset.

    Steps:
        1. Load dataset
        2. Remove PO
        3. Clean requirement text

    Returns
    -------
    pandas.DataFrame
    """

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

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
# TRAIN / TEST SPLIT
# ==========================================================

def split_dataset(df):

    """
    Create an 80/20 stratified hold-out split.

    Stratification preserves class proportions
    between training and testing sets.
    """

    print(
        "\nCreating Train/Test Split..."
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
# CATEGORY DOCUMENT CONSTRUCTION
# ==========================================================

def build_category_documents(
    train_df
):

    """
    Build one large document for each category.

    Example:

        F  -> all Functional requirements
        PE -> all Performance requirements
        SE -> all Security requirements

    IMPORTANT:
    Only TRAINING requirements are used.

    This prevents test-data leakage.
    """

    print(
        "\nBuilding Category Documents..."
    )

    category_documents = defaultdict(list)

    # ------------------------------------------------------
    # Group training requirements
    # ------------------------------------------------------

    for _, row in train_df.iterrows():

        category = row["class"]

        requirement = row["Cleaned_Text"]

        category_documents[
            category
        ].append(
            requirement
        )

    # ------------------------------------------------------
    # Concatenate requirements
    # ------------------------------------------------------

    for category in category_documents:

        category_documents[
            category
        ] = " ".join(
            category_documents[
                category
            ]
        )

    print(
        f"Generated "
        f"{len(category_documents)} "
        f"category documents."
    )

    # ------------------------------------------------------
    # Display category counts
    # ------------------------------------------------------

    for category in sorted(
        category_documents.keys()
    ):

        count = len(
            [
                row
                for _, row
                in train_df.iterrows()
                if row["class"] == category
            ]
        )

        print(
            f"{category:>3} : "
            f"{count:>4} training requirements"
        )

    return category_documents


# ==========================================================
# SECTION 4
# TF-IDF CATEGORY VECTORS
# ==========================================================

def train_category_vectors(
    category_documents
):

    """
    Train TF-IDF using the category documents.

    Configuration follows the settings shown
    in Figure 1 of the paper:

        max_df       = 0.95
        min_df       = 0.01
        max_features = 2000
        ngram_range  = (1, 2)

    IMPORTANT:
    The vectorizer is fitted ONLY on training
    category documents.
    """

    print(
        "\nTraining TF-IDF..."
    )

    # ------------------------------------------------------
    # Sort labels for deterministic ordering
    # ------------------------------------------------------

    labels = sorted(
        category_documents.keys()
    )

    documents = [
        category_documents[label]
        for label in labels
    ]

    # ------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------

    vectorizer = TfidfVectorizer(

        max_df=0.95,

        min_df=0.01,

        max_features=2000,

        ngram_range=(1, 2),

        norm="l2"

    )

    # ------------------------------------------------------
    # Fit only on training category documents
    # ------------------------------------------------------

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
# SECTION 5
# SINGLE REQUIREMENT PREDICTION
# ==========================================================

def predict_requirement(
    requirement,
    vectorizer,
    category_vectors,
    labels
):

    """
    Predict a requirement using cosine similarity.

    Algorithm:

        requirement
              |
              v
          TF-IDF
              |
              v
        cosine similarity
              |
              v
        category scores
              |
              v
        max similarity
              |
              v
        ratio threshold
              |
              v
        selected categories
              |
              v
        highest similarity
              |
              v
          prediction

    Returns
    -------
    predicted_class
    selected_categories
    scores
    max_similarity
    thresholds
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
    # COSINE SIMILARITY
    # ======================================================

    similarities = cosine_similarity(

        requirement_vector,

        category_vectors

    )[0]

    # ======================================================
    # CATEGORY SCORES
    # ======================================================

    scores = []

    for index, category in enumerate(
        labels
    ):

        score = float(
            similarities[index]
        )

        scores.append(
            (
                category,
                score
            )
        )

    # ======================================================
    # SORT BY SIMILARITY
    # ======================================================

    scores.sort(

        key=lambda x: x[1],

        reverse=True

    )

    # ======================================================
    # MAXIMUM SIMILARITY
    # ======================================================

    max_similarity = scores[0][1]

    # ======================================================
    # CATEGORY-SPECIFIC THRESHOLDS
    # ======================================================

    thresholds = {}

    selected_categories = []

    for category, score in scores:

        if category in (
            UNDERREPRESENTED_CATEGORIES
        ):

            ratio_threshold = (
                UNDERREPRESENTED_RATIO_THRESHOLD
            )

        else:

            ratio_threshold = (
                GENERAL_RATIO_THRESHOLD
            )

        threshold = (
            max_similarity
            * ratio_threshold
        )

        thresholds[
            category
        ] = threshold

        # --------------------------------------------------
        # Threshold filtering
        # --------------------------------------------------

        if score >= threshold:

            selected_categories.append(
                category
            )

    # ======================================================
    # FINAL SINGLE-LABEL PREDICTION
    # ======================================================

    # The highest-similarity category is used for
    # conventional single-label evaluation.

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
# HOLD-OUT EVALUATION
# ==========================================================

def evaluate_holdout():

    """
    Version 1

    Evaluate the classifier using an 80/20
    stratified hold-out split.
    """

    print("\n")
    print("=" * 60)
    print("HOLD-OUT EVALUATION")
    print("=" * 60)

    # ======================================================
    # DATASET
    # ======================================================

    df = prepare_dataset()

    # ======================================================
    # SPLIT
    # ======================================================

    train_df, test_df = (
        split_dataset(df)
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

    ) = train_category_vectors(
        category_documents
    )

    # ======================================================
    # PREDICTIONS
    # ======================================================

    y_true = []

    y_pred = []

    prediction_results = []

    print(
        "\nPredicting Test Requirements...\n"
    )

    total = len(test_df)

    for counter, (_, row) in enumerate(
        test_df.iterrows(),
        start=1
    ):

        actual = row["class"]

        (
            predicted,
            selected,
            scores,
            max_similarity,
            thresholds

        ) = predict_requirement(

            row["RequirementText"],

            vectorizer,

            category_vectors,

            labels

        )

        # --------------------------------------------------
        # Store labels
        # --------------------------------------------------

        y_true.append(
            actual
        )

        y_pred.append(
            predicted
        )

        # --------------------------------------------------
        # Store detailed prediction
        # --------------------------------------------------

        prediction_results.append({

            "Requirement":
                row["RequirementText"],

            "Actual":
                actual,

            "Predicted":
                predicted,

            "MaxSimilarity":
                max_similarity,

            "TopSimilarity":
                scores[0][1],

            "SelectedCategories":
                "|".join(selected),

            "Correct":
                actual == predicted

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

    macro_precision = precision_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0

    )

    macro_recall = recall_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0

    )

    macro_f1 = f1_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0

    )

    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    report_dict = classification_report(

        y_true,

        y_pred,

        labels=labels,

        output_dict=True,

        zero_division=0

    )

    report_text = classification_report(

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

    # ======================================================
    # DISPLAY PERFORMANCE
    # ======================================================

    print("\n")
    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Weighted Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{f1:.4f}"
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
    # REPORT
    # ======================================================

    print("\n")
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        report_text
    )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    print("\n")
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    cm_df = pd.DataFrame(

        cm,

        index=labels,

        columns=labels

    )

    print(
        cm_df
    )

    # ======================================================
    # SAVE PREDICTIONS
    # ======================================================

    prediction_df = pd.DataFrame(
        prediction_results
    )

    prediction_df.to_csv(

        os.path.join(
            OUTPUT_FOLDER,
            "predictions.csv"
        ),

        index=False

    )

    # ======================================================
    # SAVE CLASSIFICATION REPORT
    # ======================================================

    report_df = (
        pd.DataFrame(
            report_dict
        ).transpose()
    )

    report_df.to_csv(

        os.path.join(
            OUTPUT_FOLDER,
            "classification_report.csv"
        )

    )

    # ======================================================
    # SAVE CONFUSION MATRIX
    # ======================================================

    cm_df.to_csv(

        os.path.join(
            OUTPUT_FOLDER,
            "confusion_matrix.csv"
        )

    )

    # ======================================================
    # SAVE METRICS
    # ======================================================

    metrics_file = os.path.join(

        OUTPUT_FOLDER,

        "metrics.txt"

    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "HOLD-OUT EVALUATION\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Dataset Size : "
            f"{len(df)}\n"
        )

        file.write(
            f"Training Samples : "
            f"{len(train_df)}\n"
        )

        file.write(
            f"Testing Samples : "
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
            "General ratio = 0.80\n"
        )

        file.write(
            "L / FT ratio = 0.75\n\n"
        )

        file.write(
            f"Accuracy = "
            f"{accuracy:.6f}\n"
        )

        file.write(
            f"Weighted Precision = "
            f"{precision:.6f}\n"
        )

        file.write(
            f"Weighted Recall = "
            f"{recall:.6f}\n"
        )

        file.write(
            f"Weighted F1 = "
            f"{f1:.6f}\n"
        )

        file.write(
            f"Macro Precision = "
            f"{macro_precision:.6f}\n"
        )

        file.write(
            f"Macro Recall = "
            f"{macro_recall:.6f}\n"
        )

        file.write(
            f"Macro F1 = "
            f"{macro_f1:.6f}\n"
        )

    # ======================================================
    # FINISHED
    # ======================================================

    print("\n")
    print("=" * 60)
    print("HOLD-OUT EVALUATION COMPLETED")
    print("=" * 60)

    print(
        "\nOutput files:"
    )

    print(
        "  outputs/predictions.csv"
    )

    print(
        "  outputs/classification_report.csv"
    )

    print(
        "  outputs/confusion_matrix.csv"
    )

    print(
        "  outputs/metrics.txt"
    )

    return {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "report":
            report_dict,

        "confusion_matrix":
            cm

    }


# ==========================================================
# SECTION 7
# FIVE-FOLD CROSS VALIDATION
# ==========================================================

def evaluate_cross_validation():

    """
    Version 2

    Stratified 5-fold cross-validation.

    IMPORTANT:
    This is an experimental enhancement.

    It is NOT claimed to be the paper's
    original evaluation procedure.
    """

    print("\n")
    print("=" * 60)
    print("5-FOLD CROSS VALIDATION")
    print("=" * 60)

    df = prepare_dataset()

    X = df[
        "RequirementText"
    ]

    y = df[
        "class"
    ]

    # ======================================================
    # STRATIFIED K-FOLD
    # ======================================================

    skf = StratifiedKFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE

    )

    fold_results = []

    # ======================================================
    # FOLD LOOP
    # ======================================================

    for fold_number, (
        train_indices,
        test_indices
    ) in enumerate(

        skf.split(X, y),

        start=1

    ):

        print("\n")
        print(
            "=" * 60
        )

        print(
            f"FOLD {fold_number}/{N_SPLITS}"
        )

        print(
            "=" * 60
        )

        # --------------------------------------------------
        # Create fold datasets
        # --------------------------------------------------

        train_df = df.iloc[
            train_indices
        ].reset_index(
            drop=True
        )

        test_df = df.iloc[
            test_indices
        ].reset_index(
            drop=True
        )

        print(
            f"Training Samples : "
            f"{len(train_df)}"
        )

        print(
            f"Testing Samples  : "
            f"{len(test_df)}"
        )

        # --------------------------------------------------
        # Category documents
        # --------------------------------------------------

        category_documents = (
            build_category_documents(
                train_df
            )
        )

        # --------------------------------------------------
        # Train TF-IDF
        # --------------------------------------------------

        (
            vectorizer,
            category_vectors,
            labels

        ) = train_category_vectors(
            category_documents
        )

        # --------------------------------------------------
        # Predict
        # --------------------------------------------------

        y_true = []

        y_pred = []

        for _, row in test_df.iterrows():

            predicted, _, _, _, _ = (
                predict_requirement(

                    row["RequirementText"],

                    vectorizer,

                    category_vectors,

                    labels

                )
            )

            y_true.append(
                row["class"]
            )

            y_pred.append(
                predicted
            )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

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

        print(
            f"\nFold Accuracy : "
            f"{accuracy:.4f}"
        )

        print(
            f"Fold Precision : "
            f"{precision:.4f}"
        )

        print(
            f"Fold Recall : "
            f"{recall:.4f}"
        )

        print(
            f"Fold F1 : "
            f"{f1:.4f}"
        )

        fold_results.append({

            "Fold":
                fold_number,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1

        })

    # ======================================================
    # RESULTS
    # ======================================================

    results_df = pd.DataFrame(
        fold_results
    )

    # ------------------------------------------------------
    # Mean
    # ------------------------------------------------------

    mean_accuracy = (
        results_df["Accuracy"].mean()
    )

    mean_precision = (
        results_df["Precision"].mean()
    )

    mean_recall = (
        results_df["Recall"].mean()
    )

    mean_f1 = (
        results_df["F1"].mean()
    )

    # ------------------------------------------------------
    # Standard deviation
    # ------------------------------------------------------

    std_accuracy = (
        results_df["Accuracy"].std(
            ddof=1
        )
    )

    std_precision = (
        results_df["Precision"].std(
            ddof=1
        )
    )

    std_recall = (
        results_df["Recall"].std(
            ddof=1
        )
    )

    std_f1 = (
        results_df["F1"].std(
            ddof=1
        )
    )

    # ======================================================
    # DISPLAY
    # ======================================================

    print("\n")
    print("=" * 60)
    print("5-FOLD CROSS VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"Accuracy  : "
        f"{mean_accuracy:.4f} "
        f"+/- {std_accuracy:.4f}"
    )

    print(
        f"Precision : "
        f"{mean_precision:.4f} "
        f"+/- {std_precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{mean_recall:.4f} "
        f"+/- {std_recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{mean_f1:.4f} "
        f"+/- {std_f1:.4f}"
    )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    results_df.to_csv(

        os.path.join(
            OUTPUT_FOLDER,
            "cross_validation_results.csv"
        ),

        index=False

    )

    summary_df = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ],

        "Mean": [

            mean_accuracy,
            mean_precision,
            mean_recall,
            mean_f1

        ],

        "Std": [

            std_accuracy,
            std_precision,
            std_recall,
            std_f1

        ]

    })

    summary_df.to_csv(

        os.path.join(
            OUTPUT_FOLDER,
            "cross_validation_summary.csv"
        ),

        index=False

    )

    print("\n")
    print(
        "Saved:"
    )

    print(
        "  outputs/cross_validation_results.csv"
    )

    print(
        "  outputs/cross_validation_summary.csv"
    )

    return results_df


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # VERSION 1
    # ------------------------------------------------------

    holdout_results = (
        evaluate_holdout()
    )

    # ------------------------------------------------------
    # VERSION 2
    # ------------------------------------------------------
    #
    # Uncomment the following line when you want
    # to run the 5-fold experiment.
    #
    # This is an enhancement and is NOT claimed
    # to be the paper's original evaluation.
    # ------------------------------------------------------

    # evaluate_cross_validation()