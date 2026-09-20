import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


OUTPUT_DIR = "outputs"


def create_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_metrics(accuracy, precision, recall, f1):

    create_output_directory()

    with open(f"{OUTPUT_DIR}/metrics.txt", "w") as file:

        file.write("NLP Requirement Classification Results\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Accuracy  : {accuracy:.4f}\n")
        file.write(f"Precision : {precision:.4f}\n")
        file.write(f"Recall    : {recall:.4f}\n")
        file.write(f"F1 Score  : {f1:.4f}\n")


def save_predictions(X_test, y_test, y_pred):

    create_output_directory()

    predictions = pd.DataFrame({

        "Requirement": X_test,

        "Actual Class": y_test,

        "Predicted Class": y_pred,

        "Correct":
            ["Yes" if a == b else "No"
             for a, b in zip(y_test, y_pred)]

    })

    predictions.to_csv(
        f"{OUTPUT_DIR}/predictions.csv",
        index=False
    )


def save_classification_report(y_test, y_pred):

    create_output_directory()

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        f"{OUTPUT_DIR}/classification_report.csv"
    )


def save_confusion_matrix(y_test, y_pred, labels):

    create_output_directory()

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    cm_df.to_csv(
        f"{OUTPUT_DIR}/confusion_matrix.csv"
    )

    plt.figure(figsize=(10,8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/confusion_matrix.png",
        dpi=300
    )

    plt.close()


def save_class_distribution(df):

    create_output_directory()

    counts = df["class"].value_counts().sort_index()

    plt.figure(figsize=(10,6))

    counts.plot(kind="bar")

    plt.title("Class Distribution")

    plt.xlabel("Requirement Class")

    plt.ylabel("Number of Requirements")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/class_distribution.png",
        dpi=300
    )

    plt.close()


def generate_all_outputs(
        df,
        X_test,
        y_test,
        y_pred,
        labels,
        accuracy,
        precision,
        recall,
        f1
):

    save_metrics(
        accuracy,
        precision,
        recall,
        f1
    )

    save_predictions(
        X_test,
        y_test,
        y_pred
    )

    save_classification_report(
        y_test,
        y_pred
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        labels
    )

    save_class_distribution(df)

    print("\nAll output files generated successfully.")