"""
iteration1_reproduction.py

============================================================
ITERATION 1 - PAPER-STYLE REPRODUCTION
============================================================

Purpose
-------
Reproduce the first-iteration methodology on the original
PROMISE dataset as closely as possible using the information
available from the paper and author correspondence.

Dataset
-------
Original PROMISE NFR dataset:
    dataset/original/nfr.arff

PO category is removed, leaving 11 categories.

Method
------
1. Load PROMISE dataset
2. Remove PO
3. NLP preprocessing
4. Build one category document per NFR category
5. Train TF-IDF on the 11 category documents
6. Transform each requirement
7. Calculate cosine similarity
8. Apply category thresholds
9. If no category reaches threshold, use maximum similarity
10. Evaluate predictions

TF-IDF
-------
ngram_range = (1, 3)
min_df      = 2
max_df      = 0.95
norm        = l2

Thresholds
----------
General categories : 0.80
L and FT           : 0.75

IMPORTANT
---------
This is a PAPER-STYLE / FULL-DATA REPRODUCTION.

The category documents are constructed using the complete
PROMISE dataset before the requirements are classified.

Therefore this experiment is NOT a leakage-free hold-out
generalization evaluation.

It is intended to reproduce the first-iteration style of the
original implementation and compare its result with the
published first-iteration result.

A separate leakage-free evaluation should be reported
separately.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import nltk

from scipy.io import arff

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/original/nfr.arff"

OUTPUT_DIR = "outputs/iteration1"

GENERAL_THRESHOLD = 0.80

UNDERREPRESENTED_THRESHOLD = 0.75

UNDERREPRESENTED_CATEGORIES = {
    "L",
    "FT"
}


# ============================================================
# NLTK SETUP
# ============================================================

def initialize_nltk():

    print("=" * 70)
    print("INITIALIZING NLP RESOURCES")
    print("=" * 70)

    resources = [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab")
    ]

    for resource_name, resource_path in resources:

        try:

            nltk.data.find(resource_path)

        except LookupError:

            print(
                f"Downloading NLTK resource: "
                f"{resource_name}"
            )

            nltk.download(
                resource_name,
                quiet=True
            )

    print("NLTK resources ready.")


# ============================================================
# NLP PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Paper-style preprocessing:

    1. Lowercase
    2. Remove non-alphanumeric characters
    3. Tokenize
    4. Remove stop words
    5. Lemmatize
    """

    text = str(text).lower()

    # --------------------------------------------------------
    # Remove non-alphanumeric characters
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokens = nltk.word_tokenize(
        text
    )

    # --------------------------------------------------------
    # Stop words
    # --------------------------------------------------------

    stop_words = set(
        stopwords.words("english")
    )

    # --------------------------------------------------------
    # Lemmatizer
    # --------------------------------------------------------

    lemmatizer = WordNetLemmatizer()

    processed_tokens = []

    for token in tokens:

        if token in stop_words:
            continue

        lemma = lemmatizer.lemmatize(
            token
        )

        processed_tokens.append(
            lemma
        )

    return " ".join(
        processed_tokens
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_promise_dataset():

    print("\n")
    print("=" * 70)
    print("LOADING PROMISE DATASET")
    print("=" * 70)

    print(
        f"Path: {DATASET_PATH}"
    )

    if not os.path.exists(
        DATASET_PATH
    ):

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{DATASET_PATH}\n\n"
            f"Run this script from the "
            f"project root:\n"
            f"D:\\NLPReq"
        )

    # --------------------------------------------------------
    # Find @DATA
    # --------------------------------------------------------

    data_start = None

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for i, line in enumerate(file):

            if (
                line.strip().upper()
                == "@DATA"
            ):

                data_start = i + 1
                break

    if data_start is None:

        raise ValueError(
            "Could not find @DATA "
            "section in ARFF file."
        )

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    df = pd.read_csv(

        DATASET_PATH,

        skiprows=data_start,

        header=None,

        names=[
            "ProjectID",
            "RequirementText",
            "class"
        ],

        quotechar="'",

        skipinitialspace=True

    )

    # --------------------------------------------------------
    # Remove empty rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "RequirementText",
            "class"
        ]
    ).copy()

    # --------------------------------------------------------
    # Convert class to string
    # --------------------------------------------------------

    df["class"] = (
        df["class"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove PO
    # --------------------------------------------------------

    original_size = len(df)

    df = df[
        df["class"] != "PO"
    ].copy()

    df = df.reset_index(
        drop=True
    )

    print(
        f"Original records : "
        f"{original_size}"
    )

    print(
        f"Usable records   : "
        f"{len(df)}"
    )

    print("\nClass Distribution:")
    print(
        df["class"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# PREPROCESS DATASET
# ============================================================

def preprocess_dataset(df):

    print("\n")
    print("=" * 70)
    print("APPLYING NLP PREPROCESSING")
    print("=" * 70)

    print(
        "Lowercase"
    )

    print(
        "Remove non-alphanumeric characters"
    )

    print(
        "Tokenization"
    )

    print(
        "Stop-word removal"
    )

    print(
        "Lemmatization"
    )

    print("\nProcessing requirements...")

    df["Clean_Requirement"] = (
        df["RequirementText"]
        .apply(preprocess_text)
    )

    print(
        f"Processed requirements: "
        f"{len(df)}"
    )

    return df


# ============================================================
# BUILD CATEGORY PROFILES
# ============================================================

def build_category_profiles(df):

    print("\n")
    print("=" * 70)
    print("BUILDING CATEGORY DOCUMENTS")
    print("=" * 70)

    category_profiles = (
        df.groupby("class")[
            "Clean_Requirement"
        ]
        .apply(
            lambda values:
                " ".join(values)
        )
        .reset_index()
    )

    category_profiles.columns = [
        "Category",
        "CategoryDocument"
    ]

    category_profiles = (
        category_profiles
        .sort_values("Category")
        .reset_index(drop=True)
    )

    print(
        f"Generated "
        f"{len(category_profiles)} "
        f"category documents."
    )

    print("\nCategory Sizes:")

    category_counts = (
        df["class"]
        .value_counts()
        .sort_index()
    )

    for category in (
        category_profiles["Category"]
    ):

        count = category_counts[
            category
        ]

        print(
            f"{category:>3} : "
            f"{count:>4} requirements"
        )

    return category_profiles


# ============================================================
# TRAIN TF-IDF
# ============================================================

def train_tfidf(category_profiles):

    print("\n")
    print("=" * 70)
    print("TRAINING TF-IDF")
    print("=" * 70)

    print(
        "\nConfiguration:"
    )

    print(
        "ngram_range : (1, 3)"
    )

    print(
        "min_df      : 2"
    )

    print(
        "max_df      : 0.95"
    )

    print(
        "norm        : l2"
    )

    # --------------------------------------------------------
    # Create vectorizer
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(

        ngram_range=(1, 3),

        min_df=2,

        max_df=0.95,

        norm="l2"

    )

    documents = (
        category_profiles[
            "CategoryDocument"
        ]
        .tolist()
    )

    # --------------------------------------------------------
    # Fit TF-IDF
    # --------------------------------------------------------

    category_tfidf = (
        vectorizer.fit_transform(
            documents
        )
    )

    category_names = (
        category_profiles[
            "Category"
        ]
        .tolist()
    )

    print(
        f"\nVocabulary Size : "
        f"{len(vectorizer.vocabulary_)}"
    )

    print(
        "Category Vector Shape : "
        f"{category_tfidf.shape}"
    )

    return (
        vectorizer,
        category_tfidf,
        category_names
    )


# ============================================================
# CLASSIFY ONE REQUIREMENT
# ============================================================

def classify_requirement(

    cleaned_text,

    vectorizer,

    category_tfidf,

    category_names

):

    # --------------------------------------------------------
    # Transform requirement
    # --------------------------------------------------------

    requirement_vector = (
        vectorizer.transform(
            [cleaned_text]
        )
    )

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarities = (
        cosine_similarity(

            requirement_vector,

            category_tfidf

        )[0]
    )

    # --------------------------------------------------------
    # Store scores
    # --------------------------------------------------------

    scores = []

    for i, category in enumerate(
        category_names
    ):

        score = float(
            similarities[i]
        )

        if category in (
            UNDERREPRESENTED_CATEGORIES
        ):

            threshold = (
                UNDERREPRESENTED_THRESHOLD
            )

        else:

            threshold = (
                GENERAL_THRESHOLD
            )

        scores.append({

            "Category":
                category,

            "Similarity":
                score,

            "Threshold":
                threshold,

            "ReachedThreshold":
                score >= threshold

        })

    # --------------------------------------------------------
    # Sort highest similarity first
    # --------------------------------------------------------

    scores.sort(

        key=lambda item:
            item["Similarity"],

        reverse=True

    )

    # --------------------------------------------------------
    # Threshold selection
    # --------------------------------------------------------

    selected = [

        item

        for item in scores

        if item["ReachedThreshold"]

    ]

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if len(selected) == 0:

        selected = [
            scores[0]
        ]

        used_fallback = True

    else:

        used_fallback = False

    # --------------------------------------------------------
    # Primary prediction
    # --------------------------------------------------------

    selected.sort(

        key=lambda item:
            item["Similarity"],

        reverse=True

    )

    predicted_category = (
        selected[0]["Category"]
    )

    return (

        predicted_category,

        scores,

        used_fallback

    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate(

    df,

    vectorizer,

    category_tfidf,

    category_names

):

    print("\n")
    print("=" * 70)
    print("CLASSIFYING REQUIREMENTS")
    print("=" * 70)

    y_true = []

    y_pred = []

    prediction_rows = []

    total = len(df)

    fallback_count = 0

    # --------------------------------------------------------
    # Classification loop
    # --------------------------------------------------------

    for index, row in df.iterrows():

        predicted, scores, used_fallback = (
            classify_requirement(

                row["Clean_Requirement"],

                vectorizer,

                category_tfidf,

                category_names

            )
        )

        actual = row["class"]

        y_true.append(
            actual
        )

        y_pred.append(
            predicted
        )

        if used_fallback:

            fallback_count += 1

        # ----------------------------------------------------
        # Top similarity
        # ----------------------------------------------------

        top_score = scores[0]

        prediction_rows.append({

            "ProjectID":
                row["ProjectID"],

            "Requirement":
                row["RequirementText"],

            "Actual":
                actual,

            "Predicted":
                predicted,

            "Correct":
                actual == predicted,

            "TopCategory":
                top_score["Category"],

            "TopSimilarity":
                top_score["Similarity"],

            "TopThreshold":
                top_score["Threshold"],

            "FallbackUsed":
                used_fallback

        })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        completed = index + 1

        if (
            completed % 100 == 0
            or completed == total
        ):

            print(
                f"Processed "
                f"{completed}/{total}"
            )

    # ========================================================
    # METRICS
    # ========================================================

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

    # ========================================================
    # REPORT
    # ========================================================

    report_dict = classification_report(

        y_true,

        y_pred,

        labels=category_names,

        output_dict=True,

        zero_division=0

    )

    report_text = classification_report(

        y_true,

        y_pred,

        labels=category_names,

        zero_division=0

    )

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=category_names

    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ITERATION 1 PERFORMANCE")
    print("=" * 70)

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Weighted Precision  : "
        f"{weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall     : "
        f"{weighted_recall:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Macro Precision     : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall        : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1            : "
        f"{macro_f1:.4f}"
    )

    print(
        f"\nFallback predictions: "
        f"{fallback_count}/{total}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        report_text
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm_df = pd.DataFrame(

        cm,

        index=category_names,

        columns=category_names

    )

    print(
        cm_df
    )

    # ========================================================
    # RETURN
    # ========================================================

    metrics = {

        "Accuracy":
            accuracy,

        "WeightedPrecision":
            weighted_precision,

        "WeightedRecall":
            weighted_recall,

        "WeightedF1":
            weighted_f1,

        "MacroPrecision":
            macro_precision,

        "MacroRecall":
            macro_recall,

        "MacroF1":
            macro_f1,

        "FallbackPredictions":
            fallback_count,

        "TotalRequirements":
            total

    }

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    return (

        metrics,

        prediction_df,

        report_df,

        cm_df,

        report_text

    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(

    metrics,

    prediction_df,

    report_df,

    cm_df,

    report_text

):

    print("\n")
    print("=" * 70)
    print("SAVING ITERATION 1 RESULTS")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    prediction_path = os.path.join(

        OUTPUT_DIR,

        "predictions.csv"

    )

    prediction_df.to_csv(

        prediction_path,

        index=False

    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_path = os.path.join(

        OUTPUT_DIR,

        "classification_report.csv"

    )

    report_df.to_csv(

        report_path
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion_path = os.path.join(

        OUTPUT_DIR,

        "confusion_matrix.csv"

    )

    cm_df.to_csv(

        confusion_path
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(

        OUTPUT_DIR,

        "metrics.txt"

    )

    with open(

        metrics_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            "ITERATION 1 "
            "PAPER-STYLE REPRODUCTION\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            "Dataset: Original PROMISE\n"
        )

        file.write(
            "Usable records: 624\n"
        )

        file.write(
            "Categories: 11\n\n"
        )

        file.write(
            "Preprocessing:\n"
        )

        file.write(
            "- Lowercase\n"
        )

        file.write(
            "- Non-alphanumeric removal\n"
        )

        file.write(
            "- Tokenization\n"
        )

        file.write(
            "- Stop-word removal\n"
        )

        file.write(
            "- Lemmatization\n\n"
        )

        file.write(
            "TF-IDF:\n"
        )

        file.write(
            "- ngram_range=(1,3)\n"
        )

        file.write(
            "- min_df=2\n"
        )

        file.write(
            "- max_df=0.95\n"
        )

        file.write(
            "- norm=l2\n\n"
        )

        file.write(
            "Thresholds:\n"
        )

        file.write(
            "- General=0.80\n"
        )

        file.write(
            "- L/FT=0.75\n\n"
        )

        file.write(
            "Evaluation type:\n"
        )

        file.write(
            "Paper-style full-data "
            "reproduction.\n"
        )

        file.write(
            "This is NOT a leakage-free "
            "hold-out evaluation.\n\n"
        )

        file.write(
            "Metrics:\n"
        )

        for key, value in metrics.items():

            if isinstance(
                value,
                float
            ):

                file.write(
                    f"{key}: "
                    f"{value:.6f}\n"
                )

            else:

                file.write(
                    f"{key}: "
                    f"{value}\n"
                )

        file.write(
            "\n\nClassification Report\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            report_text
        )

    print(
        "\nGenerated:"
    )

    print(
        f"  {prediction_path}"
    )

    print(
        f"  {report_path}"
    )

    print(
        f"  {confusion_path}"
    )

    print(
        f"  {metrics_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "ITERATION 1 "
        "PAPER-STYLE REPRODUCTION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # NLTK
    # --------------------------------------------------------

    initialize_nltk()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_promise_dataset()

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    df = preprocess_dataset(
        df
    )

    # --------------------------------------------------------
    # Build category profiles
    # --------------------------------------------------------

    category_profiles = (
        build_category_profiles(
            df
        )
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    (
        vectorizer,
        category_tfidf,
        category_names

    ) = train_tfidf(
        category_profiles
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    (
        metrics,
        prediction_df,
        report_df,
        cm_df,
        report_text

    ) = evaluate(

        df,

        vectorizer,

        category_tfidf,

        category_names

    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(

        metrics,

        prediction_df,

        report_df,

        cm_df,

        report_text

    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ITERATION 1 REPRODUCTION COMPLETED")
    print("=" * 70)

    print(
        "\nFinal Weighted F1 : "
        f"{metrics['WeightedF1']:.4f}"
    )

    print(
        "Final Accuracy     : "
        f"{metrics['Accuracy']:.4f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is the paper-style "
        "full-data reproduction."
    )

    print(
        "It should NOT be described "
        "as unseen-test accuracy."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()