"""
evaluation_expanded.py

Independent Dataset Expansion Experiment

Dataset:
    outputs/expanded_dataset.csv

Methodology:
    1. Load expanded dataset
    2. Preprocess requirements
    3. Stratified 80/20 train-test split
    4. Build one category document per class using TRAINING data only
    5. Train TF-IDF using TRAINING category documents only
    6. Generate one TF-IDF vector per category
    7. Calculate cosine similarity between each test requirement
       and every category vector
    8. Apply the same threshold-based classification methodology
    9. Calculate Accuracy, Precision, Recall, F1
   10. Generate classification report
   11. Generate confusion matrix

IMPORTANT:
This is an independent dataset-expansion experiment.
It is NOT the original Iteration 2 or Iteration 3 dataset
from the IEEE paper.
"""

# ============================================================
# IMPORTS
# ============================================================

import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
# Change this:
# from sklearn.metrics import cosine_similarity, ...

# To this:
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "outputs/expanded_dataset.csv"

OUTPUT_FOLDER = "outputs/expanded"

RANDOM_STATE = 42

TEST_SIZE = 0.20

# Same threshold strategy used by your category classifier
THRESHOLD_RATIO = 0.80

# Your 11-category experiment
CATEGORIES = [
    "A",
    "F",
    "FT",
    "L",
    "LF",
    "MN",
    "O",
    "PE",
    "SC",
    "SE",
    "US"
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 70)
    print("LOADING EXPANDED DATASET")
    print("=" * 70)

    if not os.path.exists(DATASET_FILE):

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATASET_FILE}\n\n"
            "Run merge_datasets.py first."
        )

    df = pd.read_csv(
        DATASET_FILE
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "RequirementText",
        "class"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "RequirementText",
            "class"
        ]
    ).copy()

    # --------------------------------------------------------
    # Remove PO
    # --------------------------------------------------------

    df = df[
        df["class"] != "PO"
    ].copy()

    # --------------------------------------------------------
    # Keep only our 11 classes
    # --------------------------------------------------------

    df = df[
        df["class"].isin(CATEGORIES)
    ].copy()

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    print(
        f"\nDataset Size : {len(df)}"
    )

    print(
        "\nClass Distribution:"
    )

    print(
        df["class"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_dataset(df):

    print()
    print("=" * 70)
    print("APPLYING NLP PREPROCESSING")
    print("=" * 70)

    print(
        "\nPreprocessing requirements..."
    )

    df = df.copy()

    df["cleaned_text"] = (
        df["RequirementText"]
        .apply(clean_text)
    )

    # Remove requirements that become empty
    df = df[
        df["cleaned_text"].str.strip() != ""
    ].copy()

    df = df.reset_index(
        drop=True
    )

    print(
        f"Requirements after preprocessing : {len(df)}"
    )

    return df


# ============================================================
# BUILD CATEGORY DOCUMENTS
# ============================================================

def build_category_documents(
    train_df
):

    print()
    print("=" * 70)
    print("BUILDING CATEGORY DOCUMENTS")
    print("=" * 70)

    category_documents = {}

    for category in CATEGORIES:

        category_requirements = train_df[
            train_df["class"] == category
        ]["cleaned_text"].tolist()

        # ----------------------------------------------------
        # Concatenate all training requirements belonging
        # to this category into one document.
        # ----------------------------------------------------

        category_document = " ".join(
            category_requirements
        )

        category_documents[
            category
        ] = category_document

        print(
            f"{category:>3} : "
            f"{len(category_requirements):>4} "
            f"training requirements"
        )

    return category_documents


# ============================================================
# TRAIN TF-IDF
# ============================================================

def train_tfidf(
    category_documents
):

    print()
    print("=" * 70)
    print("TRAINING TF-IDF")
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # TF-IDF is fitted ONLY on training category documents.
    #
    # This prevents test information from entering the model.
    # --------------------------------------------------------

    documents = [
        category_documents[category]
        for category in CATEGORIES
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        norm="l2"
    )

    category_matrix = vectorizer.fit_transform(
        documents
    )

    print(
        f"\nVocabulary Size : "
        f"{len(vectorizer.vocabulary_)}"
    )

    print(
        f"Category Vector Shape : "
        f"{category_matrix.shape}"
    )

    return (
        vectorizer,
        category_matrix
    )


# ============================================================
# PREDICT ONE REQUIREMENT
# ============================================================

def predict_requirement(
    requirement,
    vectorizer,
    category_matrix
):

    # --------------------------------------------------------
    # Convert requirement into TF-IDF vector
    # --------------------------------------------------------

    requirement_vector = vectorizer.transform(
        [requirement]
    )

    # --------------------------------------------------------
    # Compare requirement with every category vector
    # --------------------------------------------------------

    similarities = cosine_similarity(
        requirement_vector,
        category_matrix
    )[0]

    # --------------------------------------------------------
    # Store scores
    # --------------------------------------------------------

    scores = {}

    for index, category in enumerate(CATEGORIES):

        scores[
            category
        ] = float(
            similarities[index]
        )

    # --------------------------------------------------------
    # Find maximum similarity
    # --------------------------------------------------------

    max_similarity = max(
        scores.values()
    )

    # --------------------------------------------------------
    # Threshold
    #
    # A category is selected when:
    #
    # similarity >=
    # maximum_similarity * 0.80
    #
    # --------------------------------------------------------

    threshold = (
        max_similarity
        * THRESHOLD_RATIO
    )

    selected_categories = [
        category
        for category in CATEGORIES
        if scores[category] >= threshold
    ]

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if not selected_categories:

        best_category = max(
            scores,
            key=scores.get
        )

        selected_categories = [
            best_category
        ]

    # --------------------------------------------------------
    # For evaluation we need ONE predicted class.
    #
    # If multiple categories pass the threshold,
    # choose the category having the highest similarity.
    #
    # This is an implementation assumption necessary
    # to calculate standard single-label classification
    # metrics against the PROMISE dataset.
    # --------------------------------------------------------

    predicted_category = max(
        selected_categories,
        key=lambda category: scores[category]
    )

    return (
        predicted_category,
        scores,
        max_similarity,
        threshold,
        selected_categories
    )


# ============================================================
# EVALUATE TEST DATA
# ============================================================

def evaluate_model(
    test_df,
    vectorizer,
    category_matrix
):

    print()
    print("=" * 70)
    print("PREDICTING TEST REQUIREMENTS")
    print("=" * 70)

    y_true = []

    y_pred = []

    prediction_records = []

    total = len(test_df)

    for counter, (_, row) in enumerate(
        test_df.iterrows(),
        start=1
    ):

        actual_class = row["class"]

        cleaned_requirement = row[
            "cleaned_text"
        ]

        (
            predicted_category,
            scores,
            max_similarity,
            threshold,
            selected_categories
        ) = predict_requirement(

            cleaned_requirement,

            vectorizer,

            category_matrix

        )

        y_true.append(
            actual_class
        )

        y_pred.append(
            predicted_category
        )

        # ----------------------------------------------------
        # Save prediction information
        # ----------------------------------------------------

        record = {

            "RequirementText":
                row["RequirementText"],

            "CleanedText":
                cleaned_requirement,

            "ActualClass":
                actual_class,

            "PredictedClass":
                predicted_category,

            "MaximumSimilarity":
                max_similarity,

            "Threshold":
                threshold,

            "SelectedCategories":
                "|".join(
                    selected_categories
                )

        }

        # Add every category score
        for category in CATEGORIES:

            record[
                f"Similarity_{category}"
            ] = scores[category]

        prediction_records.append(
            record
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if counter % 100 == 0 or counter == total:

            print(
                f"Processed "
                f"{counter}/{total}"
            )

    return (
        y_true,
        y_pred,
        prediction_records
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

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

    return {

        "Accuracy":
            accuracy,

        "Weighted Precision":
            precision,

        "Weighted Recall":
            recall,

        "Weighted F1":
            f1,

        "Macro Precision":
            macro_precision,

        "Macro Recall":
            macro_recall,

        "Macro F1":
            macro_f1

    }


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    y_true,
    y_pred
):

    report = classification_report(

        y_true,

        y_pred,

        labels=CATEGORIES,

        target_names=CATEGORIES,

        zero_division=0,

        output_dict=True

    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    output_file = os.path.join(

        OUTPUT_FOLDER,

        "classification_report.csv"

    )

    report_df.to_csv(
        output_file
    )

    return report_df


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=CATEGORIES

    )

    cm_df = pd.DataFrame(

        cm,

        index=CATEGORIES,

        columns=CATEGORIES

    )

    output_file = os.path.join(

        OUTPUT_FOLDER,

        "confusion_matrix.csv"

    )

    cm_df.to_csv(
        output_file
    )

    return cm_df


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    prediction_records
):

    predictions_df = pd.DataFrame(
        prediction_records
    )

    output_file = os.path.join(

        OUTPUT_FOLDER,

        "predictions.csv"

    )

    predictions_df.to_csv(

        output_file,

        index=False,

        encoding="utf-8"

    )

    return predictions_df


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics,
    train_size,
    test_size,
    vocabulary_size
):

    output_file = os.path.join(

        OUTPUT_FOLDER,

        "metrics.txt"

    )

    with open(

        output_file,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            "INDEPENDENT DATASET EXPANSION "
            "EXPERIMENT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "Dataset: PROMISE + Promise+\n"
        )

        file.write(
            "Final dataset size: 3529\n"
        )

        file.write(
            "Test size: 20%\n"
        )

        file.write(
            "Random state: 42\n"
        )

        file.write(
            f"Training samples: {train_size}\n"
        )

        file.write(
            f"Testing samples: {test_size}\n"
        )

        file.write(
            f"Vocabulary size: {vocabulary_size}\n"
        )

        file.write(
            f"Threshold ratio: "
            f"{THRESHOLD_RATIO}\n\n"
        )

        file.write(
            "NOTE:\n"
        )

        file.write(
            "This is an independent dataset-expansion "
            "experiment and is NOT the original "
            "Iteration 2 or Iteration 3 dataset "
            "from the IEEE paper.\n\n"
        )

        for name, value in metrics.items():

            file.write(
                f"{name:<22}: "
                f"{value:.4f}\n"
            )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    metrics,
    report_df,
    confusion_df
):

    print()
    print("=" * 70)
    print("EXPANDED DATASET MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"Accuracy           : "
        f"{metrics['Accuracy']:.4f}"
    )

    print(
        f"Weighted Precision : "
        f"{metrics['Weighted Precision']:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{metrics['Weighted Recall']:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{metrics['Weighted F1']:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{metrics['Macro Precision']:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{metrics['Macro Recall']:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{metrics['Macro F1']:.4f}"
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        report_df.to_string()
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        confusion_df.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("INDEPENDENT DATASET EXPANSION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. Preprocess
    # --------------------------------------------------------

    df = preprocess_dataset(
        df
    )

    # --------------------------------------------------------
    # 3. Train/Test Split
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING TRAIN/TEST SPLIT")
    print("=" * 70)

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
        f"\nTraining Samples : "
        f"{len(train_df)}"
    )

    print(
        f"Testing Samples  : "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Check classes in training data
    # --------------------------------------------------------

    print(
        "\nTraining Class Distribution:"
    )

    print(
        train_df["class"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTesting Class Distribution:"
    )

    print(
        test_df["class"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 4. Build category documents
    # --------------------------------------------------------

    category_documents = (
        build_category_documents(
            train_df
        )
    )

    # --------------------------------------------------------
    # 5. Train TF-IDF
    # --------------------------------------------------------

    (
        vectorizer,
        category_matrix
    ) = train_tfidf(
        category_documents
    )

    # --------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------

    (
        y_true,
        y_pred,
        prediction_records
    ) = evaluate_model(

        test_df,

        vectorizer,

        category_matrix

    )

    # --------------------------------------------------------
    # 7. Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(

        y_true,

        y_pred

    )

    # --------------------------------------------------------
    # 8. Classification report
    # --------------------------------------------------------

    report_df = (
        save_classification_report(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # 9. Confusion matrix
    # --------------------------------------------------------

    confusion_df = (
        save_confusion_matrix(
            y_true,
            y_pred
        )
    )

    # --------------------------------------------------------
    # 10. Predictions
    # --------------------------------------------------------

    save_predictions(
        prediction_records
    )

    # --------------------------------------------------------
    # 11. Metrics file
    # --------------------------------------------------------

    save_metrics(

        metrics,

        len(train_df),

        len(test_df),

        len(vectorizer.vocabulary_)

    )

    # --------------------------------------------------------
    # 12. Print results
    # --------------------------------------------------------

    print_results(

        metrics,

        report_df,

        confusion_df

    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPANDED DATASET EVALUATION COMPLETED")
    print("=" * 70)

    print()
    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT_FOLDER}"
    )

    print()

    print(
        "Generated files:"
    )

    print(
        "  metrics.txt"
    )

    print(
        "  predictions.csv"
    )

    print(
        "  classification_report.csv"
    )

    print(
        "  confusion_matrix.csv"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()