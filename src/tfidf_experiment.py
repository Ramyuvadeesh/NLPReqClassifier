"""
tfidf_experiment.py

TF-IDF CONFIGURATION ABLATION EXPERIMENT
========================================

Purpose
-------
Determine whether TF-IDF configuration is limiting the performance
of the category-document cosine-similarity classifier.

IMPORTANT
---------
This is an experimental ablation.

It does NOT claim that all configurations below were used in the
IEEE paper.

The paper's methodology remains our reference implementation.

Only the TF-IDF configuration is changed between experiments.

FIXED
-----
Dataset          : PROMISE
PO               : removed
Split            : 80/20 stratified
Random state     : 42
Representation   : One concatenated document per category
Similarity       : Cosine similarity
Prediction       : Highest cosine similarity
Threshold        : 0.80 general / 0.75 L and FT

VARIABLE
--------
TF-IDF parameters:

1. Unigrams, unlimited vocabulary
2. Unigrams + bigrams, unlimited vocabulary
3. Unigrams + bigrams + trigrams, unlimited vocabulary
4. Unigrams + bigrams, 5,000 features
5. Unigrams + bigrams, 10,000 features
6. Unigrams + bigrams, 2,000 features
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os

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

GENERAL_RATIO = 0.80

UNDERREPRESENTED_RATIO = 0.75

UNDERREPRESENTED_CATEGORIES = {
    "L",
    "FT"
}


# ==========================================================
# TF-IDF CONFIGURATIONS
# ==========================================================

TFIDF_CONFIGURATIONS = [

    {
        "Name": "unigram_unlimited",
        "ngram_range": (1, 1),
        "max_features": None,
        "min_df": 1,
        "max_df": 0.95
    },

    {
        "Name": "bigram_unlimited",
        "ngram_range": (1, 2),
        "max_features": None,
        "min_df": 1,
        "max_df": 0.95
    },

    {
        "Name": "trigram_unlimited",
        "ngram_range": (1, 3),
        "max_features": None,
        "min_df": 1,
        "max_df": 0.95
    },

    {
        "Name": "bigram_5000",
        "ngram_range": (1, 2),
        "max_features": 5000,
        "min_df": 1,
        "max_df": 0.95
    },

    {
        "Name": "bigram_10000",
        "ngram_range": (1, 2),
        "max_features": 10000,
        "min_df": 1,
        "max_df": 0.95
    },

    {
        "Name": "bigram_2000",
        "ngram_range": (1, 2),
        "max_features": 2000,
        "min_df": 0.01,
        "max_df": 0.95
    }
]


# ==========================================================
# DATASET
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

    print(
        "Preprocessing completed."
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

    return (
        train_df,
        test_df
    )


# ==========================================================
# CATEGORY DOCUMENTS
# ==========================================================

def build_category_documents(train_df):

    print(
        "\nBuilding Category Documents..."
    )

    category_documents = defaultdict(list)

    for _, row in train_df.iterrows():

        category = row["class"]

        text = row["Cleaned_Text"]

        category_documents[
            category
        ].append(text)

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

def train_tfidf(
    category_documents,
    configuration
):

    vectorizer = TfidfVectorizer(

        ngram_range=
            configuration["ngram_range"],

        max_features=
            configuration["max_features"],

        min_df=
            configuration["min_df"],

        max_df=
            configuration["max_df"],

        norm="l2"

    )

    labels = sorted(
        category_documents.keys()
    )

    documents = [

        category_documents[label]

        for label in labels

    ]

    category_vectors = (
        vectorizer.fit_transform(
            documents
        )
    )

    return (
        vectorizer,
        category_vectors,
        labels
    )


# ==========================================================
# PREDICTION
# ==========================================================

def predict_requirement(

    requirement,

    vectorizer,

    category_vectors,

    labels

):

    cleaned = clean_text(
        requirement
    )

    requirement_vector = (
        vectorizer.transform(
            [cleaned]
        )
    )

    # ------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------

    similarities = cosine_similarity(

        requirement_vector,

        category_vectors

    )[0]

    # ------------------------------------------------------
    # Rank categories
    # ------------------------------------------------------

    scores = list(
        zip(
            labels,
            similarities
        )
    )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # ------------------------------------------------------
    # Highest similarity
    # ------------------------------------------------------

    max_similarity = scores[0][1]

    # ------------------------------------------------------
    # Threshold
    # ------------------------------------------------------

    selected = []

    for category, score in scores:

        if category in (
            UNDERREPRESENTED_CATEGORIES
        ):

            ratio = (
                UNDERREPRESENTED_RATIO
            )

        else:

            ratio = GENERAL_RATIO

        threshold = (
            max_similarity * ratio
        )

        if score >= threshold:

            selected.append(
                category
            )

    # ------------------------------------------------------
    # Single-label prediction
    # ------------------------------------------------------

    predicted = scores[0][0]

    return (
        predicted,
        selected,
        scores,
        max_similarity
    )


# ==========================================================
# EVALUATE ONE CONFIGURATION
# ==========================================================

def evaluate_configuration(

    configuration,

    train_df,

    test_df,

    category_documents

):

    print("\n")
    print("=" * 70)
    print(
        "CONFIGURATION:",
        configuration["Name"]
    )
    print("=" * 70)

    print(
        "ngram_range :",
        configuration["ngram_range"]
    )

    print(
        "max_features :",
        configuration["max_features"]
    )

    print(
        "min_df :",
        configuration["min_df"]
    )

    print(
        "max_df :",
        configuration["max_df"]
    )

    # ------------------------------------------------------
    # Train TF-IDF
    # ------------------------------------------------------

    (
        vectorizer,
        category_vectors,
        labels

    ) = train_tfidf(

        category_documents,

        configuration

    )

    vocabulary_size = (
        len(
            vectorizer.vocabulary_
        )
    )

    print(
        "\nVocabulary Size :",
        vocabulary_size
    )

    print(
        "Category Vector Shape :",
        category_vectors.shape
    )

    # ------------------------------------------------------
    # Predictions
    # ------------------------------------------------------

    y_true = []

    y_pred = []

    prediction_results = []

    total = len(test_df)

    print(
        "\nPredicting test requirements..."
    )

    for counter, (_, row) in enumerate(

        test_df.iterrows(),

        start=1

    ):

        (
            predicted,
            selected,
            scores,
            max_similarity

        ) = predict_requirement(

            row["RequirementText"],

            vectorizer,

            category_vectors,

            labels

        )

        actual = row["class"]

        y_true.append(
            actual
        )

        y_pred.append(
            predicted
        )

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
                )

        })

        if (
            counter % 100 == 0
            or counter == total
        ):

            print(
                f"Processed "
                f"{counter}/{total}"
            )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

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

    print("\n")
    print(
        f"Vocabulary Size : "
        f"{vocabulary_size}"
    )

    print(
        f"Accuracy         : "
        f"{accuracy:.4f}"
    )

    print(
        f"Weighted F1      : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Macro F1         : "
        f"{macro_f1:.4f}"
    )

    return {

        "Configuration":
            configuration["Name"],

        "Ngram":
            str(
                configuration["ngram_range"]
            ),

        "MaxFeatures":
            configuration["max_features"],

        "MinDF":
            configuration["min_df"],

        "MaxDF":
            configuration["max_df"],

        "VocabularySize":
            vocabulary_size,

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
            macro_f1

    }


# ==========================================================
# MAIN EXPERIMENT
# ==========================================================

def run_experiment():

    print("\n")
    print("=" * 70)
    print("TF-IDF CONFIGURATION ABLATION EXPERIMENT")
    print("=" * 70)

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    df = prepare_dataset()

    # ------------------------------------------------------
    # Same fixed split
    # ------------------------------------------------------

    train_df, test_df = create_split(
        df
    )

    # ------------------------------------------------------
    # Category documents
    # ------------------------------------------------------

    category_documents = (
        build_category_documents(
            train_df
        )
    )

    # ------------------------------------------------------
    # Run configurations
    # ------------------------------------------------------

    results = []

    for configuration in (
        TFIDF_CONFIGURATIONS
    ):

        result = evaluate_configuration(

            configuration,

            train_df,

            test_df,

            category_documents

        )

        results.append(
            result
        )

    # ------------------------------------------------------
    # Results dataframe
    # ------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # ------------------------------------------------------
    # Sort by accuracy
    # ------------------------------------------------------

    results_sorted = (
        results_df.sort_values(

            by="Accuracy",

            ascending=False

        )
    )

    # ------------------------------------------------------
    # Display
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TF-IDF EXPERIMENT RESULTS")
    print("=" * 70)

    display_columns = [

        "Configuration",

        "Ngram",

        "MaxFeatures",

        "MinDF",

        "VocabularySize",

        "Accuracy",

        "WeightedPrecision",

        "WeightedRecall",

        "WeightedF1",

        "MacroF1"

    ]

    print(

        results_sorted[
            display_columns
        ].to_string(
            index=False
        )

    )

    # ------------------------------------------------------
    # Best configuration
    # ------------------------------------------------------

    best = (
        results_sorted.iloc[0]
    )

    print("\n")
    print("=" * 70)
    print("BEST TF-IDF CONFIGURATION")
    print("=" * 70)

    print(
        f"Configuration : "
        f"{best['Configuration']}"
    )

    print(
        f"Vocabulary    : "
        f"{best['VocabularySize']}"
    )

    print(
        f"Accuracy      : "
        f"{best['Accuracy']:.4f}"
    )

    print(
        f"Weighted F1   : "
        f"{best['WeightedF1']:.4f}"
    )

    print(
        f"Macro F1      : "
        f"{best['MacroF1']:.4f}"
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_path = os.path.join(

        OUTPUT_FOLDER,

        "tfidf_experiment.csv"

    )

    results_sorted.to_csv(

        output_path,

        index=False

    )

    print("\n")
    print(
        "Results saved to:"
    )

    print(
        f"  {output_path}"
    )

    print("\n")
    print("=" * 70)
    print("TF-IDF EXPERIMENT COMPLETED")
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    run_experiment()