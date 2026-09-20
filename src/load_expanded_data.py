import pandas as pd


EXPANDED_DATASET = "outputs/expanded_dataset.csv"


def load_expanded_dataset():

    df = pd.read_csv(
        EXPANDED_DATASET
    )

    # Make sure required columns exist
    required_columns = [
        "ProjectID",
        "RequirementText",
        "class"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    # Remove accidental PO records
    df = df[
        df["class"] != "PO"
    ].copy()

    # Remove rows with missing values
    df = df.dropna(
        subset=[
            "RequirementText",
            "class"
        ]
    )

    # Reset index
    df = df.reset_index(
        drop=True
    )

    print(
        "Expanded Dataset Loaded"
    )

    print(
        "Number of requirements:",
        len(df)
    )

    print(
        "\nClass Distribution:"
    )

    print(
        df["class"].value_counts()
    )

    return df


if __name__ == "__main__":

    df = load_expanded_dataset()