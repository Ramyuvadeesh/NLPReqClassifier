"""
merge_datasets.py

Creates an expanded software-requirements dataset by combining:

1. Original PROMISE NFR dataset
2. Promise+ dataset

The original PROMISE records are kept as the authoritative records
when an identical requirement occurs in both datasets.

PO (Portability) is removed because the current project evaluates
the 11 classes used by the classifier.

Outputs:

    outputs/
        dataset_report.txt
        duplicate_requirements.csv
        conflicting_requirements.csv
        expanded_dataset.csv
"""

import os
import re
import csv

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ORIGINAL_FILE = "dataset/original/nfr.arff"

PROMISE_PLUS_FILE = "dataset/expansion/Promise+.arff"

OUTPUT_FOLDER = "outputs"

EXPANDED_FILE = os.path.join(
    OUTPUT_FOLDER,
    "expanded_dataset.csv"
)

DUPLICATE_FILE = os.path.join(
    OUTPUT_FOLDER,
    "duplicate_requirements.csv"
)

CONFLICT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "conflicting_requirements.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "dataset_report.txt"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize requirement text for duplicate detection.

    This does NOT change the actual requirement stored
    in the final dataset.

    It is only used to determine whether two requirements
    are effectively the same.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove surrounding whitespace
    text = text.strip()

    return text


# ============================================================
# ARFF READER
# ============================================================

def read_arff(file_path):
    """
    Read an ARFF file containing:

        ProjectID
        RequirementText
        class

    The function is intentionally implemented here instead
    of depending on scipy because the two datasets use slightly
    different ARFF formatting.
    """

    print()
    print("=" * 70)
    print("Reading:")
    print(file_path)
    print("=" * 70)

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    rows = []

    data_started = False

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        for raw_line in file:

            line = raw_line.strip()

            # ------------------------------------------------
            # Find @DATA
            # ------------------------------------------------

            if line.upper() == "@DATA":

                data_started = True

                continue

            if not data_started:

                continue

            # Ignore blank lines
            if not line:

                continue

            # Ignore comments
            if line.startswith("%"):

                continue

            # ------------------------------------------------
            # Parse:
            #
            # ProjectID,'Requirement text',CLASS
            #
            # ------------------------------------------------

            try:

                # First comma separates ProjectID
                first_comma = line.find(",")

                if first_comma == -1:

                    continue

                project_id = line[
                    :first_comma
                ].strip()

                remaining = line[
                    first_comma + 1:
                ].strip()

                # ------------------------------------------------
                # Requirement is enclosed in single quotes
                # ------------------------------------------------

                if not remaining.startswith("'"):

                    continue

                # Find the closing quote followed by the class.
                #
                # We search from the end because requirement text
                # itself may contain commas.
                # ------------------------------------------------

                last_quote = remaining.rfind("'")

                if last_quote <= 0:

                    continue

                requirement = remaining[
                    1:last_quote
                ]

                class_label = remaining[
                    last_quote + 1:
                ].strip()

                # Remove possible comma
                class_label = class_label.lstrip(",")

                # Remove accidental whitespace
                class_label = class_label.strip()

                if not requirement:

                    continue

                if not class_label:

                    continue

                rows.append({

                    "ProjectID": project_id,

                    "RequirementText": requirement,

                    "class": class_label

                })

            except Exception as error:

                print(
                    "Warning: Could not parse line:"
                )

                print(line)

                print("Reason:", error)

    df = pd.DataFrame(rows)

    print(
        f"Records loaded: {len(df)}"
    )

    return df


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary(
    df,
    name
):
    """
    Print basic dataset information.
    """

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(
        "Number of records:",
        len(df)
    )

    print()

    print(
        "Class distribution:"
    )

    print(
        df["class"].value_counts()
    )


# ============================================================
# FIND DUPLICATES BETWEEN DATASETS
# ============================================================

def find_cross_dataset_duplicates(
    original_df,
    promise_plus_df
):
    """
    Find requirements appearing in both datasets.
    """

    original_map = {}

    for _, row in original_df.iterrows():

        key = normalize_text(
            row["RequirementText"]
        )

        if key:

            original_map.setdefault(
                key,
                []
            ).append(row)

    duplicate_rows = []

    conflict_rows = []

    for _, row in promise_plus_df.iterrows():

        key = normalize_text(
            row["RequirementText"]
        )

        if key not in original_map:

            continue

        for original_row in original_map[key]:

            duplicate_rows.append({

                "Original_ProjectID":
                    original_row["ProjectID"],

                "Original_Class":
                    original_row["class"],

                "PromisePlus_ProjectID":
                    row["ProjectID"],

                "PromisePlus_Class":
                    row["class"],

                "RequirementText":
                    row["RequirementText"]

            })

            # --------------------------------------------
            # Same requirement but different labels
            # --------------------------------------------

            if (
                original_row["class"]
                !=
                row["class"]
            ):

                conflict_rows.append({

                    "Original_ProjectID":
                        original_row["ProjectID"],

                    "Original_Class":
                        original_row["class"],

                    "PromisePlus_ProjectID":
                        row["ProjectID"],

                    "PromisePlus_Class":
                        row["class"],

                    "RequirementText":
                        row["RequirementText"]

                })

    duplicates_df = pd.DataFrame(
        duplicate_rows
    )

    conflicts_df = pd.DataFrame(
        conflict_rows
    )

    return (
        duplicates_df,
        conflicts_df
    )


# ============================================================
# BUILD EXPANDED DATASET
# ============================================================

def build_expanded_dataset(
    original_df,
    promise_plus_df
):
    """
    Build the expanded dataset.

    Rules:

    1. Keep original PROMISE records.
    2. Remove PO.
    3. Add only Promise+ requirements that do not
       already exist in PROMISE.
    4. Do not automatically resolve conflicting labels.
       Conflicts are excluded and reported separately.
    """

    # --------------------------------------------------------
    # Remove PO
    # --------------------------------------------------------

    original_df = original_df[
        original_df["class"] != "PO"
    ].copy()

    promise_plus_df = promise_plus_df[
        promise_plus_df["class"] != "PO"
    ].copy()

    # --------------------------------------------------------
    # Normalize text
    # --------------------------------------------------------

    original_df["_normalized"] = (
        original_df["RequirementText"]
        .apply(normalize_text)
    )

    promise_plus_df["_normalized"] = (
        promise_plus_df["RequirementText"]
        .apply(normalize_text)
    )

    # --------------------------------------------------------
    # Get original PROMISE requirements
    # --------------------------------------------------------

    original_texts = set(
        original_df["_normalized"]
    )

    # --------------------------------------------------------
    # Find Promise+ requirements not already in PROMISE
    # --------------------------------------------------------

    new_records = []

    for _, row in promise_plus_df.iterrows():

        normalized = row["_normalized"]

        if normalized not in original_texts:

            new_records.append({

                "ProjectID":
                    row["ProjectID"],

                "RequirementText":
                    row["RequirementText"],

                "class":
                    row["class"],

                "Source":
                    "Promise+"

            })

    new_df = pd.DataFrame(
        new_records
    )

    # --------------------------------------------------------
    # Add source information to original data
    # --------------------------------------------------------

    original_output = original_df[
        [
            "ProjectID",
            "RequirementText",
            "class"
        ]
    ].copy()

    original_output["Source"] = "PROMISE"

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    expanded_df = pd.concat(

        [
            original_output,
            new_df
        ],

        ignore_index=True

    )

    # --------------------------------------------------------
    # Remove helper columns if present
    # --------------------------------------------------------

    expanded_df = expanded_df[
        [
            "ProjectID",
            "RequirementText",
            "class",
            "Source"
        ]
    ]

    # --------------------------------------------------------
    # Final duplicate safety check
    # --------------------------------------------------------

    expanded_df["_normalized"] = (
        expanded_df["RequirementText"]
        .apply(normalize_text)
    )

    expanded_df = expanded_df.drop_duplicates(
        subset=["_normalized"],
        keep="first"
    )

    expanded_df = expanded_df.drop(
        columns=["_normalized"]
    )

    expanded_df = expanded_df.reset_index(
        drop=True
    )

    return expanded_df


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
    original_df,
    promise_plus_df,
    duplicates_df,
    conflicts_df,
    expanded_df
):

    original_without_po = original_df[
        original_df["class"] != "PO"
    ]

    promise_plus_without_po = promise_plus_df[
        promise_plus_df["class"] != "PO"
    ]

    report_lines = []

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        "DATASET EXPANSION REPORT"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append("")

    report_lines.append(
        f"Original PROMISE records: "
        f"{len(original_df)}"
    )

    report_lines.append(
        f"Original PROMISE usable records "
        f"(PO removed): "
        f"{len(original_without_po)}"
    )

    report_lines.append(
        f"Promise+ records: "
        f"{len(promise_plus_df)}"
    )

    report_lines.append(
        f"Promise+ usable records "
        f"(PO removed): "
        f"{len(promise_plus_without_po)}"
    )

    report_lines.append("")

    report_lines.append(
        f"Cross-dataset duplicate matches: "
        f"{len(duplicates_df)}"
    )

    report_lines.append(
        f"Conflicting-label matches: "
        f"{len(conflicts_df)}"
    )

    report_lines.append("")

    report_lines.append(
        f"Final expanded dataset: "
        f"{len(expanded_df)}"
    )

    report_lines.append("")

    report_lines.append(
        "FINAL CLASS DISTRIBUTION"
    )

    report_lines.append(
        "-" * 70
    )

    distribution = (
        expanded_df["class"]
        .value_counts()
        .sort_index()
    )

    for label, count in distribution.items():

        percentage = (
            count /
            len(expanded_df)
            * 100
        )

        report_lines.append(

            f"{label:>3} : "
            f"{count:>4} "
            f"({percentage:>6.2f}%)"

        )

    report_lines.append("")

    report_lines.append(
        "SOURCE DISTRIBUTION"
    )

    report_lines.append(
        "-" * 70
    )

    source_distribution = (
        expanded_df["Source"]
        .value_counts()
    )

    for source, count in source_distribution.items():

        report_lines.append(

            f"{source:>12} : "
            f"{count}"

        )

    report_lines.append("")

    report_lines.append(
        "NOTE"
    )

    report_lines.append(
        "This expanded dataset is an independent "
        "dataset-expansion experiment."
    )

    report_lines.append(
        "It must NOT be described as the original "
        "Iteration 2 or Iteration 3 dataset "
        "from the IEEE paper."
    )

    report_lines.append(
        "The original later-iteration datasets "
        "were not available for redistribution."
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(report_lines)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PROMISE DATASET EXPANSION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    original_df = read_arff(
        ORIGINAL_FILE
    )

    promise_plus_df = read_arff(
        PROMISE_PLUS_FILE
    )

    # --------------------------------------------------------
    # Print summaries
    # --------------------------------------------------------

    print_dataset_summary(
        original_df,
        "ORIGINAL PROMISE"
    )

    print_dataset_summary(
        promise_plus_df,
        "PROMISE+"
    )

    # --------------------------------------------------------
    # Find duplicates
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Comparing datasets...")
    print("=" * 70)

    (
        duplicates_df,
        conflicts_df
    ) = find_cross_dataset_duplicates(

        original_df,

        promise_plus_df

    )

    print()
    print(
        "Duplicate matches:",
        len(duplicates_df)
    )

    print(
        "Conflicting labels:",
        len(conflicts_df)
    )

    # --------------------------------------------------------
    # Save duplicate report
    # --------------------------------------------------------

    if len(duplicates_df) > 0:

        duplicates_df.to_csv(

            DUPLICATE_FILE,

            index=False,

            encoding="utf-8"

        )

    else:

        pd.DataFrame(
            columns=[
                "Original_ProjectID",
                "Original_Class",
                "PromisePlus_ProjectID",
                "PromisePlus_Class",
                "RequirementText"
            ]
        ).to_csv(

            DUPLICATE_FILE,

            index=False

        )

    # --------------------------------------------------------
    # Save conflict report
    # --------------------------------------------------------

    if len(conflicts_df) > 0:

        conflicts_df.to_csv(

            CONFLICT_FILE,

            index=False,

            encoding="utf-8"

        )

    else:

        pd.DataFrame(
            columns=[
                "Original_ProjectID",
                "Original_Class",
                "PromisePlus_ProjectID",
                "PromisePlus_Class",
                "RequirementText"
            ]
        ).to_csv(

            CONFLICT_FILE,

            index=False

        )

    # --------------------------------------------------------
    # Build expanded dataset
    # --------------------------------------------------------

    expanded_df = build_expanded_dataset(

        original_df,

        promise_plus_df

    )

    # --------------------------------------------------------
    # Save expanded dataset
    # --------------------------------------------------------

    expanded_df.to_csv(

        EXPANDED_FILE,

        index=False,

        encoding="utf-8"

    )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    generate_report(

        original_df,

        promise_plus_df,

        duplicates_df,

        conflicts_df,

        expanded_df

    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPANSION COMPLETED")
    print("=" * 70)

    print()

    print(
        f"Final dataset size: "
        f"{len(expanded_df)}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        f"  {EXPANDED_FILE}"
    )

    print(
        f"  {DUPLICATE_FILE}"
    )

    print(
        f"  {CONFLICT_FILE}"
    )

    print(
        f"  {REPORT_FILE}"
    )

    print()

    print(
        "Final class distribution:"
    )

    print(
        expanded_df["class"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()