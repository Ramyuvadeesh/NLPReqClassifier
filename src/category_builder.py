"""
category_builder.py

Purpose:
--------
Groups all software requirements into their corresponding
requirement categories and creates one large document for
each category.

This implements the "Group Text by Category" step shown
in Figure 2 of the IEEE paper.
"""

from collections import defaultdict

from load_data import load_dataset
from preprocessing import clean_text


def build_category_documents():
    """
    Groups all requirements into category documents.

    Returns
    -------
    dict
        {
            'F': 'requirement requirement requirement ...',
            'SE': 'requirement requirement requirement ...',
            ...
        }
    """

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = load_dataset()

    # Remove the PO class (only one sample)
    df = df[df["class"] != "PO"].reset_index(drop=True)

    print(f"Dataset Size : {len(df)}")

    print("\nPreprocessing Requirements...")

    df["Cleaned_Text"] = df["RequirementText"].apply(clean_text)

    # Dictionary to store category documents
    category_documents = defaultdict(list)

    print("\nGrouping Requirements by Category...\n")

    # Append each cleaned requirement to its category
    for _, row in df.iterrows():

        category = row["class"]

        requirement = row["Cleaned_Text"]

        category_documents[category].append(requirement)

    # Merge all requirements of a category into one document
    for category in category_documents:

        category_documents[category] = " ".join(
            category_documents[category]
        )

    return category_documents


def display_statistics(category_documents):
    """
    Displays statistics about category documents.
    """

    print("=" * 60)
    print("CATEGORY DOCUMENT STATISTICS")
    print("=" * 60)

    for category, document in sorted(category_documents.items()):

        word_count = len(document.split())

        print(f"{category:>3} : {word_count:5} words")


def preview_documents(category_documents):
    """
    Prints the first 300 characters
    of every category document.
    """

    print("\n")
    print("=" * 60)
    print("CATEGORY DOCUMENT PREVIEW")
    print("=" * 60)

    for category, document in sorted(category_documents.items()):

        print(f"\nCategory : {category}")

        print("-" * 60)

        print(document[:300] + "...")

        print()


if __name__ == "__main__":

    category_documents = build_category_documents()

    display_statistics(category_documents)

    preview_documents(category_documents)