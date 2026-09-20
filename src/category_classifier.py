"""
category_classifier.py

Threshold-based software requirement classifier.

Methodology confirmed by the paper author:
    - TF-IDF representation
    - Cosine similarity
    - General similarity threshold = 0.80
    - Temporary threshold = 0.75 for underrepresented
      categories such as Legal (L) and Fault Tolerance (FT)

IMPORTANT:
The thresholds are implemented as absolute cosine-similarity
thresholds:

    General categories -> 0.80
    L and FT           -> 0.75

If no category reaches its configured threshold, the category
with the highest cosine similarity is used as a fallback.

This classifier is intended for individual requirement
prediction/demo.
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import joblib

from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FOLDER = "models"

VECTORIZER_FILE = os.path.join(
    MODEL_FOLDER,
    "tfidf_vectorizer.pkl"
)

CATEGORY_VECTORS_FILE = os.path.join(
    MODEL_FOLDER,
    "category_vectors.pkl"
)

CATEGORY_LABELS_FILE = os.path.join(
    MODEL_FOLDER,
    "category_labels.pkl"
)


# ============================================================
# THRESHOLDS
# ============================================================

GENERAL_THRESHOLD = 0.80

UNDERREPRESENTED_THRESHOLD = 0.75

UNDERREPRESENTED_CATEGORIES = {
    "L",
    "FT"
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading trained model...")

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    category_vectors = joblib.load(
        CATEGORY_VECTORS_FILE
    )

    category_labels = joblib.load(
        CATEGORY_LABELS_FILE
    )

    print("Model Loaded Successfully.")

    return (
        vectorizer,
        category_vectors,
        category_labels
    )


# ============================================================
# GET CATEGORY THRESHOLD
# ============================================================

def get_category_threshold(category):
    """
    Return the absolute cosine-similarity threshold
    for a category.
    """

    if category in UNDERREPRESENTED_CATEGORIES:

        return UNDERREPRESENTED_THRESHOLD

    return GENERAL_THRESHOLD


# ============================================================
# CLASSIFY REQUIREMENT
# ============================================================

def classify_requirement(
    requirement,
    vectorizer,
    category_vectors,
    category_labels
):

    # ========================================================
    # PREPROCESS
    # ========================================================

    cleaned_requirement = clean_text(
        requirement
    )

    # ========================================================
    # TF-IDF VECTOR
    # ========================================================

    requirement_vector = (
        vectorizer.transform(
            [cleaned_requirement]
        )
    )

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    similarity_vector = (
        cosine_similarity(
            requirement_vector,
            category_vectors
        )[0]
    )

    # ========================================================
    # STORE CATEGORY SCORES
    # ========================================================

    scores = {}

    for index, category in enumerate(
        category_labels
    ):

        scores[category] = float(
            similarity_vector[index]
        )

    # ========================================================
    # RANK CATEGORIES
    # ========================================================

    ranked_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # ========================================================
    # APPLY ABSOLUTE THRESHOLDS
    # ========================================================

    selected_categories = []

    category_thresholds = {}

    for category in category_labels:

        threshold = get_category_threshold(
            category
        )

        category_thresholds[category] = (
            threshold
        )

        similarity = scores[category]

        if similarity >= threshold:

            selected_categories.append(
                category
            )

    # ========================================================
    # FALLBACK
    # ========================================================

    fallback_used = False

    if not selected_categories:

        fallback_used = True

        highest_category = (
            ranked_scores[0][0]
        )

        selected_categories.append(
            highest_category
        )

    # ========================================================
    # MAXIMUM SIMILARITY
    # ========================================================

    max_similarity = ranked_scores[0][1]

    return (
        cleaned_requirement,
        scores,
        ranked_scores,
        selected_categories,
        max_similarity,
        category_thresholds,
        fallback_used
    )


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    cleaned_requirement,
    scores,
    ranked_scores,
    selected_categories,
    max_similarity,
    category_thresholds,
    fallback_used
):

    # ========================================================
    # CLEANED REQUIREMENT
    # ========================================================

    print()
    print("=" * 60)
    print("CLEANED REQUIREMENT")
    print("=" * 60)

    print(
        cleaned_requirement
    )

    # ========================================================
    # CATEGORY SCORES
    # ========================================================

    print()
    print("=" * 60)
    print("CATEGORY SCORES")
    print("=" * 60)

    for category, score in ranked_scores:

        threshold = category_thresholds[
            category
        ]

        print(
            f"{category:>3} : "
            f"{score:.4f} "
            f"(threshold={threshold:.2f})"
        )

    # ========================================================
    # MAXIMUM SIMILARITY
    # ========================================================

    print()
    print(
        f"Maximum Similarity : "
        f"{max_similarity:.6f}"
    )

    # ========================================================
    # SELECTED CATEGORIES
    # ========================================================

    print()
    print("=" * 60)
    print("SELECTED CATEGORIES")
    print("=" * 60)

    for category in selected_categories:
        print(
            f"{category:>3} : "
            f"{scores[category]:.4f}"
        )

    # ========================================================
    # FALLBACK INFORMATION
    # ========================================================

    if fallback_used:

        print()
        print(
            "NOTE: No category reached "
            "its absolute cosine threshold."
        )

        print(
            "Fallback: highest-similarity "
            "category selected."
        )

    else:

        print()
        print(
            "At least one category reached "
            "its configured threshold."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        vectorizer,
        category_vectors,
        category_labels
    ) = load_model()

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    print(
        "\nEnter Requirement:\n"
    )

    requirement = input().strip()

    if not requirement:

        print(
            "No requirement entered."
        )

        return

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    (
        cleaned_requirement,
        scores,
        ranked_scores,
        selected_categories,
        max_similarity,
        category_thresholds,
        fallback_used

    ) = classify_requirement(

        requirement,

        vectorizer,

        category_vectors,

        category_labels

    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_result(

        cleaned_requirement,

        scores,

        ranked_scores,

        selected_categories,

        max_similarity,

        category_thresholds,

        fallback_used

    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()