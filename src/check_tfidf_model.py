import os
import joblib


MODEL_FILE = "models/tfidf_vectorizer.pkl"


def main():

    print("=" * 70)
    print("SAVED TF-IDF MODEL INSPECTION")
    print("=" * 70)

    if not os.path.exists(MODEL_FILE):
        print("\nERROR:")
        print(f"Model not found: {MODEL_FILE}")
        return

    vectorizer = joblib.load(MODEL_FILE)

    print("\nTF-IDF PARAMETERS")
    print("-" * 70)

    print(
        f"ngram_range       : "
        f"{vectorizer.ngram_range}"
    )

    print(
        f"min_df            : "
        f"{vectorizer.min_df}"
    )

    print(
        f"max_df            : "
        f"{vectorizer.max_df}"
    )

    print(
        f"max_features      : "
        f"{vectorizer.max_features}"
    )

    print(
        f"norm              : "
        f"{vectorizer.norm}"
    )

    print(
        f"sublinear_tf      : "
        f"{vectorizer.sublinear_tf}"
    )

    print(
        f"use_idf           : "
        f"{vectorizer.use_idf}"
    )

    print(
        f"smooth_idf        : "
        f"{vectorizer.smooth_idf}"
    )

    print(
        f"lowercase         : "
        f"{vectorizer.lowercase}"
    )

    print(
        f"Vocabulary size   : "
        f"{len(vectorizer.vocabulary_)}"
    )

    print("\n")
    print("=" * 70)
    print("EXPECTED ITERATION 1 CONFIGURATION")
    print("=" * 70)

    print("ngram_range       : (1, 3)")
    print("min_df            : 2")
    print("max_df            : 0.95")
    print("max_features      : None")
    print("norm              : l2")

    print("\n")
    print("=" * 70)
    print("CONFIGURATION CHECK")
    print("=" * 70)

    expected = {
        "ngram_range": (1, 3),
        "min_df": 2,
        "max_df": 0.95,
        "max_features": None,
        "norm": "l2"
    }

    checks = []

    for parameter, expected_value in expected.items():

        actual_value = getattr(
            vectorizer,
            parameter
        )

        passed = (
            actual_value == expected_value
        )

        checks.append(passed)

        status = "PASS" if passed else "FAIL"

        print(
            f"{parameter:<18} "
            f"{status:<6} "
            f"actual={actual_value} "
            f"expected={expected_value}"
        )

    print("\n")

    if all(checks):

        print(
            "RESULT: TF-IDF configuration "
            "matches the Iteration 1 configuration."
        )

    else:

        print(
            "RESULT: TF-IDF configuration "
            "DOES NOT match the Iteration 1 configuration."
        )

    print("\n")
    print("=" * 70)


if __name__ == "__main__":
    main()