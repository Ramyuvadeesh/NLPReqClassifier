# NLPReqClassifier — Software Requirements Classification Using NLP

A Python-based mini-project that automatically classifies software requirement statements into **11 requirement categories** using Natural Language Processing (NLP), **TF-IDF vectorization**, and **cosine similarity**.

The project uses the **PROMISE Non-Functional Requirements (NFR) dataset**, expands its experiments with synthetic requirement samples, evaluates predictions over four iterations, and provides a **command-line live prediction demo**.

## Features

- Preprocesses requirement text using lowercasing, punctuation removal, tokenization, stop-word removal, and lemmatization.
- Builds a text profile for each requirement category.
- Represents category profiles and input requirements using TF-IDF.
- Predicts the most relevant category using cosine similarity and category-specific thresholds.
- Uses a maximum-similarity fallback when no category meets its threshold.
- Evaluates predictions with accuracy, precision, recall, F1-score, classification reports, and confusion matrices.
- Generates CSV reports and visualizations.
- Allows users to type their own requirements for interactive classification.

## Requirement Categories

| Code | Category |
| --- | --- |
| `F` | Functional |
| `A` | Availability |
| `FT` | Fault Tolerance |
| `L` | Legal |
| `LF` | Look and Feel |
| `MN` | Maintainability |
| `O` | Operational |
| `PE` | Performance |
| `SC` | Scalability |
| `SE` | Security |
| `US` | Usability |

The `PO` category from the original dataset is excluded by the loading code.

## Technologies Used

- **Language:** Python
- **Data processing:** Pandas, NumPy, SciPy
- **NLP:** NLTK
- **Text features and evaluation:** scikit-learn
- **Plots:** Matplotlib, Seaborn
- **Dataset format:** ARFF

## Project Structure

```text
mini_project/
├── dataset/
│   └── original/
│       ├── nfr.arff
│       ├── synthetic_test_113.arff
│       ├── synthetic_test_89.arff
│       ├── synthetic_test_81.arff
│       └── synthetic_test_77_erp.arff
├── src/
│   ├── iteration1_reproduction.py
│   ├── iteration2_reproduction.py
│   ├── iteration3_reproducion.py
│   ├── final_iteration.py
│   ├── live_demo.py
│   ├── generate_synthetic_113.py
│   ├── generate_synthetic_89.py
│   ├── generate_synthetic_81.py
│   └── generate_synthetic_77.py
├── outputs/
│   └── iteration1/
│       ├── metrics.txt
│       ├── metrics1.txt
│       ├── metrics2.txt
│       ├── metrics3.txt
│       ├── metrics4.txt
│       ├── classification_report.csv
│       ├── confusion_matrix.csv
│       ├── predictions.csv
│       ├── confusion_matrix.png
│       ├── confusion_matrix1.png
│       ├── confusion_matrix2.png
│       ├── confusion_matrix3.png
│       ├── confusion_matrix4.png
│       └── metrics_bar_graph.png
└── README.md
```

> **Note:** The filename `iteration3_reproducion.py` is spelled exactly this way in the uploaded source code.

## Installation

**1. Get the project**

Download and extract the project ZIP or clone your GitHub repository. Open a terminal **inside the `mini_project` folder** (the folder containing `dataset`, `src`, and `outputs`).

**2. (Optional) Create a virtual environment**

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

**3. Install the dependencies**

```bash
python -m pip install numpy pandas scipy nltk scikit-learn matplotlib seaborn
```

The Python scripts attempt to download missing NLTK resources (`stopwords`, `wordnet`, `punkt`, and `punkt_tab`) at startup. An internet connection may be needed on first run.

## How to Run

Run the following commands **from the `mini_project` root directory** so the relative dataset paths resolve correctly.

### Interactive requirement classification

```bash
python src/live_demo.py
```

The live demo trains category profiles using the original PROMISE data and the four included synthetic datasets, then prompts you to enter a software requirement.

Example input:

```text
The system must encrypt all user passwords in the database.
```

The program displays a **predicted category** and its **cosine similarity score**. Type `exit` or `quit` to stop.

### Run the evaluation iterations

```bash
python src/iteration1_reproduction.py
python src/iteration2_reproduction.py
python src/iteration3_reproducion.py
python src/final_iteration.py
```

| Script | Training data | Test data |
| --- | --- | --- |
| Iteration 1 | Original PROMISE dataset (624 usable requirements) | 113 synthetic requirements |
| Iteration 2 | Original + 113 synthetic requirements (737) | 89 synthetic requirements |
| Iteration 3 | Original + 113 + 89 synthetic requirements (826) | 81 synthetic requirements |
| Final iteration | Original + 113 + 89 + 81 synthetic requirements (907) | 77 synthetic ERP-style requirements |

### How classification works

1. Load requirement statements and their categories from ARFF files.
2. Normalize and preprocess the text using NLTK.
3. Combine the training statements of each category into a category profile.
4. Calculate TF-IDF vectors for category profiles and incoming requirements.
5. Compare them using cosine similarity.
6. Select the highest-scoring category that meets its threshold, or use the highest score as a fallback.
7. For evaluation runs, compare predictions against the test labels and save metrics and plots.

The classifier uses a general similarity threshold of **0.80**, reduced to **0.75** for the underrepresented `L` and `FT` categories. TF-IDF settings differ between the evaluation scripts and the live demo.

## Results

The included `outputs/iteration1/metrics*.txt` files contain the following **recorded results on synthetic test sets**:

| Evaluation | Test samples | Accuracy | Weighted F1-score |
| --- | ---: | ---: | ---: |
| Iteration 1 | 113 | 81.42% | 81.89% |
| Iteration 2 | 89 | 84.27% | 84.67% |
| Iteration 3 | 81 | 95.06% | 95.06% |
| Final iteration | 77 | 89.61% | 90.18% |

These figures are taken from the saved metrics files, **not from a fresh execution**. Because the tests use generated synthetic requirements, they should not be interpreted as independently verified performance on real-world requirements. Some saved metrics descriptions are also reused across iterations; the test-set sizes identify the corresponding runs.

### Example output visualizations

**Confusion matrix**

![Saved confusion matrix](outputs/iteration1/confusion_matrix.png)

**Metrics comparison chart**

![Saved metrics bar graph](outputs/iteration1/metrics_bar_graph.png)

Evaluation scripts write reports, predictions, and visualizations under `outputs/iteration1/`. **Running several scripts successively can overwrite shared output filenames**; rename or copy output folders to preserve separate runs.

## Synthetic Data Generation

The `generate_synthetic_*.py` scripts can generate synthetic ARFF datasets for experimentation.

For example:

```bash
python src/generate_synthetic_113.py
```

**Important:** These generator scripts currently save files directly under `dataset/`, while the evaluation and demo scripts read from `dataset/original/`. If you regenerate data, move the generated ARFF files into `dataset/original/` or update the corresponding paths in the scripts. Generated datasets may also differ between runs because the scripts use random sampling.

## Scope and Limitations

- This is a **command-line research/academic mini-project**, not a deployed website or API.
- It implements **category-profile similarity classification**; it does not train a neural network.
- The provided evaluations use **synthetic hold-out test sets**, which may not reflect real-world performance.
- The live demo trains on all included datasets for interactive prediction; it is **not a separate hold-out evaluation**.
- Evaluation scripts reuse the same output directory and contain a few legacy log messages, so check the actual input files and metrics when comparing runs.

## Future Improvements

- Test on an independently collected set of real software requirements.
- Add more robust train/test separation and repeatable experiments.
- Create a web interface or REST API for requirement classification.
- Save the fitted vectorizer and category profiles so the live demo does not retrain on every launch.
- Organize per-iteration outputs into separate directories.

## Project Purpose

This mini-project explores how traditional NLP techniques can assist **software requirements engineering** by reducing the manual effort involved in categorizing requirement statements.
