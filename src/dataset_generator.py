"""
dataset_generator.py

Targeted AI Synthetic Dataset Generator
========================================

Purpose
-------
Generate an independent synthetic training dataset for Iteration 2.

IMPORTANT EXPERIMENTAL RULES
-----------------------------
1. The original PROMISE dataset is split FIRST using the fixed
   80/20 stratified split (random_state=42).
2. The original test set is NEVER given to the AI model.
3. Synthetic requirements are added ONLY to the original training set.
4. Synthetic records are deduplicated against the ENTIRE original
   PROMISE dataset and against each other.
5. Synthetic data is an independent experiment and must NOT be
   described as the original paper's Iteration 2 dataset.
6. The final generated CSV keeps the PROMISE-compatible columns:
       RequirementText,class
   Metadata is written to a separate audit CSV.

Iteration 2 quota
-----------------
MN 30
US 30
A  25
LF 20
L  15
SC 15
SE 5
O  5
FT 5
PE 0
F  0

Total = 150

OpenAI API
----------
Set:
    OPENAI_API_KEY=...

Optional:
    OPENAI_MODEL=gpt-5.6-luna

Install:
    pip install openai pandas scikit-learn

Run:
    python src/dataset_generator.py

Dry run (creates prompts without calling the API):
    python src/dataset_generator.py --dry-run

The model is asked to return JSON:
{
  "requirements": [
      "The system shall ..."
  ]
}
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_FILE = ROOT / "dataset" / "original" / "nfr.arff"

GENERATED_DIR = ROOT / "dataset" / "generated"
SPLIT_DIR = ROOT / "dataset" / "splits"
ITERATION_DIR = ROOT / "dataset" / "iterations"

GENERATED_FILE = GENERATED_DIR / "iteration2_generated.csv"
AUDIT_FILE = GENERATED_DIR / "iteration2_generated_audit.csv"
TRAIN_FILE = SPLIT_DIR / "iteration1_train.csv"
TEST_FILE = SPLIT_DIR / "iteration1_test.csv"
ITERATION2_TRAIN_FILE = ITERATION_DIR / "iteration2_train.csv"

GENERATED_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
ITERATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

VALID_CATEGORIES = [
    "A", "F", "FT", "L", "LF", "MN",
    "O", "PE", "SC", "SE", "US"
]

QUOTAS = {
    "MN": 30,
    "US": 30,
    "A": 25,
    "LF": 20,
    "L": 15,
    "SC": 15,
    "SE": 5,
    "O": 5,
    "FT": 5,
    "PE": 0,
    "F": 0,
}

# Change this environment variable if a different model is desired.
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")


# ============================================================
# TARGETED GENERATION SPECIFICATION
# ============================================================

CATEGORY_SPECS = {
    "MN": {
        "definition": (
            "Maintainability: requirements concerning how easily a software "
            "product can be maintained, updated, modified, repaired, configured, "
            "or supported over its lifecycle."
        ),
        "subtopics": [
            "scheduled maintenance windows",
            "software updates and upgrades",
            "maintenance cost or budget constraints",
            "configuration and repair activities",
            "ease of modifying or maintaining software",
            "maintenance scheduling",
        ],
        "boundary": (
            "Avoid turning the requirement into Performance (PE), where the "
            "primary concern is response time/throughput, or Scalability (SC), "
            "where the primary concern is workload/capacity growth."
        ),
    },
    "US": {
        "definition": (
            "Usability: requirements concerning ease of learning, ease of use, "
            "user effectiveness, user satisfaction, navigation simplicity, "
            "user assistance, accessibility from the user's perspective, "
            "and reduction of user effort or error."
        ),
        "subtopics": [
            "learnability",
            "navigation simplicity",
            "user satisfaction",
            "user assistance/help",
            "user training",
            "user effectiveness",
            "preventing user mistakes",
            "accessible interaction",
        ],
        "boundary": (
            "Avoid requirements whose primary concern is visual appearance "
            "(LF), a concrete system function (F), legal compliance (L), "
            "security/access control (SE), or raw response time (PE)."
        ),
    },
    "A": {
        "definition": (
            "Availability: requirements concerning the proportion of time a "
            "system or service remains available and operational, uptime, "
            "downtime limits, continuity of service, and availability targets."
        ),
        "subtopics": [
            "uptime percentage",
            "maximum permitted downtime",
            "service availability windows",
            "continuous service",
            "availability during peak periods",
            "service continuity targets",
        ],
        "boundary": (
            "Do not use the word 'available' merely to mean that a file, "
            "feature, or product can be accessed/downloaded. The primary "
            "concern must be service/system availability."
        ),
    },
    "LF": {
        "definition": (
            "Look & Feel: requirements concerning visual presentation and "
            "interface appearance, including layout, colors, typography, "
            "graphics, visual consistency, and aesthetic presentation."
        ),
        "subtopics": [
            "screen layout",
            "visual consistency",
            "colors and typography",
            "icons and graphics",
            "visual hierarchy",
            "interface appearance",
            "presentation style",
        ],
        "boundary": (
            "Avoid making the requirement primarily about what the system "
            "does (F) or how easy it is for users to learn/use it (US)."
        ),
    },
    "L": {
        "definition": (
            "Legal: requirements concerning laws, regulations, standards "
            "with legal force, statutory obligations, regulatory compliance, "
            "contractual/legal constraints, or mandatory legal rules."
        ),
        "subtopics": [
            "regulatory compliance",
            "statutory requirements",
            "privacy/data protection laws",
            "financial regulations",
            "industry regulations",
            "records retention mandated by law",
            "accessibility laws when framed as legal compliance",
        ],
        "boundary": (
            "The primary reason for the requirement must be a legal or "
            "regulatory obligation, not merely good security, usability, "
            "interface design, or normal system functionality."
        ),
    },
    "SC": {
        "definition": (
            "Scalability: requirements concerning the ability of a system "
            "to accommodate increases in users, transactions, workload, "
            "data volume, or system capacity while continuing to operate."
        ),
        "subtopics": [
            "concurrent users",
            "transaction volume growth",
            "data volume growth",
            "capacity expansion",
            "horizontal scaling",
            "vertical scaling",
            "future workload growth",
        ],
        "boundary": (
            "Avoid making response-time the primary concern (PE). The main "
            "concern must be growth or capacity."
        ),
    },
    "SE": {
        "definition": (
            "Security: requirements concerning protection of systems and "
            "data, authentication, authorization, access control, confidentiality, "
            "integrity, and prevention of unauthorized access."
        ),
        "subtopics": [
            "authentication",
            "authorization",
            "role-based access",
            "confidentiality",
            "integrity protection",
            "unauthorized access prevention",
        ],
        "boundary": (
            "Keep these distinct from ordinary Functional requirements. "
            "The security property must be the primary concern."
        ),
    },
    "O": {
        "definition": (
            "Operability: requirements concerning operating, deploying, "
            "installing, configuring, administering, interfacing with, or "
            "running the software in its intended operational environment."
        ),
        "subtopics": [
            "installation",
            "deployment",
            "configuration",
            "operating-system compatibility",
            "system administration",
            "operational procedures",
            "integration with operational infrastructure",
        ],
        "boundary": (
            "Avoid pure maintenance, performance, or scalability requirements."
        ),
    },
    "FT": {
        "definition": (
            "Fault Tolerance: requirements concerning continued or safe "
            "operation when faults, component failures, service failures, "
            "or error conditions occur."
        ),
        "subtopics": [
            "continued operation after component failure",
            "failure recovery",
            "redundancy",
            "failover",
            "graceful degradation",
            "fault isolation",
        ],
        "boundary": (
            "The requirement must primarily concern behavior under failure, "
            "not ordinary availability during normal operation."
        ),
    },
}


# ============================================================
# PROMPTS
# ============================================================

BASE_SYSTEM_PROMPT = """
You generate labeled software requirements for a research experiment in
Non-Functional Requirement (NFR) classification.

The requirements will be used ONLY as synthetic TRAINING DATA.

Rules:
- Write realistic, concise software requirements.
- Prefer the form: "The system/product/application shall ..."
- Each requirement must express one clear primary requirement concern.
- Do not copy, paraphrase, or closely imitate known PROMISE dataset sentences.
- Do not invent citations, standards, organizations, or laws that sound like
  factual references unless the prompt explicitly asks for a generic example.
- Use varied software domains such as healthcare, banking, education,
  e-commerce, logistics, HR, transportation, media, manufacturing, and
  enterprise systems.
- Vary wording, sentence length, actors, metrics, and domain vocabulary.
- Do not repeatedly use the same keywords.
- Do not use category names such as "maintainability", "usability",
  "availability", "security", etc. merely to reveal the label.
- The requirement must be classifiable from its actual meaning.
- Do not generate multi-label requirements. Each item must have one primary
  target category.
- Return valid JSON only.
""".strip()


def build_prompt(category: str, count: int) -> str:
    spec = CATEGORY_SPECS[category]

    subtopics = "\n".join(
        f"- {item}" for item in spec["subtopics"]
    )

    return f"""
Generate exactly {count} UNIQUE synthetic software requirements for category "{category}".

Category definition:
{spec["definition"]}

Required topic coverage:
{subtopics}

Boundary guidance:
{spec["boundary"]}

Additional diversity requirements:
- Use at least 5 different application domains.
- Mix quantified and non-quantified requirements.
- Mix short and medium-length requirements.
- Do not make every sentence start with exactly the same phrase.
- Do not simply replace nouns in one template.
- Avoid near-duplicates.
- Avoid copying phrases from the PROMISE dataset.
- Each requirement must have ONE primary category: {category}.

Return exactly this JSON structure:
{{
  "requirements": [
    "requirement 1",
    "requirement 2"
  ]
}}
""".strip()


# ============================================================
# DATA LOADING
# ============================================================

def read_original_arff(path: Path) -> pd.DataFrame:
    """
    Read the PROMISE ARFF using the project's established format:
    ProjectID, RequirementText, class.
    """
    rows = []
    data_started = False

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()

            if line.upper() == "@DATA":
                data_started = True
                continue

            if not data_started or not line or line.startswith("%"):
                continue

            # Robust CSV parsing for quoted requirement text.
            import csv

            try:
                parts = next(csv.reader(
                    [line],
                    quotechar="'",
                    skipinitialspace=True
                ))
            except Exception:
                continue

            if len(parts) < 3:
                continue

            rows.append({
                "ProjectID": parts[0].strip(),
                "RequirementText": parts[1].strip(),
                "class": parts[-1].strip(),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No records could be read from {path}")

    return df


# ============================================================
# NORMALIZATION / DUPLICATES
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# FIXED ORIGINAL SPLIT
# ============================================================

def create_fixed_split(df: pd.DataFrame):
    df = df[df["class"].isin(VALID_CATEGORIES)].copy()
    df = df.dropna(subset=["RequirementText", "class"]).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["class"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(TRAIN_FILE, index=False, encoding="utf-8")
    test_df.to_csv(TEST_FILE, index=False, encoding="utf-8")

    print(f"Original training set : {len(train_df)}")
    print(f"Original test set     : {len(test_df)}")
    print(f"Saved                 : {TRAIN_FILE}")
    print(f"Saved                 : {TEST_FILE}")

    return train_df, test_df


# ============================================================
# OPENAI GENERATION
# ============================================================

def get_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI package is not installed. Run: pip install openai"
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it before generation."
        )

    return OpenAI()


def parse_json_response(text: str):
    text = text.strip()

    # Remove accidental Markdown fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")

    payload = json.loads(text[start:end + 1])

    requirements = payload.get("requirements")

    if not isinstance(requirements, list):
        raise ValueError("JSON does not contain a requirements list.")

    return [
        str(item).strip()
        for item in requirements
        if str(item).strip()
    ]


def generate_category(client, category: str, count: int, max_attempts=4):
    prompt = build_prompt(category, count)

    for attempt in range(1, max_attempts + 1):
        print(f"\nGenerating {count} {category} requirements "
              f"(attempt {attempt}/{max_attempts})...")

        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "system",
                    "content": BASE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        requirements = parse_json_response(response.output_text)

        # Remove exact duplicates within this batch.
        unique = []
        seen = set()

        for requirement in requirements:
            key = normalize_text(requirement)
            if key and key not in seen:
                seen.add(key)
                unique.append(requirement)

        if len(unique) == count:
            return unique

        print(
            f"Model returned {len(unique)} unique items; "
            f"expected {count}. Retrying."
        )

        prompt = build_prompt(category, count)
        time.sleep(1)

    raise RuntimeError(
        f"Could not obtain exactly {count} unique requirements for {category}."
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_generated(generated_df: pd.DataFrame, original_df: pd.DataFrame):
    if generated_df["class"].isna().any():
        raise ValueError("Generated data contains missing class labels.")

    if not set(generated_df["class"]).issubset(set(VALID_CATEGORIES)):
        raise ValueError("Generated data contains unsupported classes.")

    original_keys = set(
        original_df["RequirementText"].map(normalize_text)
    )

    generated_keys = generated_df["RequirementText"].map(normalize_text)

    duplicated_against_original = generated_keys.isin(original_keys)

    if duplicated_against_original.any():
        bad = generated_df.loc[duplicated_against_original, "RequirementText"]
        raise ValueError(
            "Generated data contains requirements duplicated from PROMISE:\n"
            + "\n".join(bad.tolist()[:10])
        )

    if generated_keys.duplicated().any():
        raise ValueError("Generated dataset contains duplicate requirements.")

    expected_counts = pd.Series(QUOTAS).sort_index()
    actual_counts = (
        generated_df["class"]
        .value_counts()
        .reindex(expected_counts.index, fill_value=0)
        .sort_index()
    )

    if not actual_counts.equals(expected_counts):
        raise ValueError(
            "Generated class counts do not match requested quotas.\n"
            f"Expected:\n{expected_counts}\n"
            f"Actual:\n{actual_counts}"
        )


# ============================================================
# BUILD ITERATION 2 TRAINING SET
# ============================================================

def build_iteration2_train(original_train: pd.DataFrame,
                           generated: pd.DataFrame):
    original_train = original_train.copy()
    generated = generated.copy()

    original_train["Source"] = "PROMISE"
    generated["Source"] = "AI_Synthetic"
    generated["Iteration"] = "Iteration2"

    original_train["Iteration"] = "Iteration2"

    combined = pd.concat(
        [
            original_train[
                ["ProjectID", "RequirementText", "class", "Source", "Iteration"]
            ],
            generated[
                ["ProjectID", "RequirementText", "class", "Source", "Iteration"]
            ],
        ],
        ignore_index=True,
    )

    combined.to_csv(
        ITERATION2_TRAIN_FILE,
        index=False,
        encoding="utf-8",
    )

    print(f"\nIteration 2 training size: {len(combined)}")
    print(combined["class"].value_counts().sort_index())
    print(f"Saved: {ITERATION2_TRAIN_FILE}")

    return combined


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts and create the fixed split without calling the API.",
    )
    args = parser.parse_args()

    print("=" * 75)
    print("ITERATION 2 TARGETED SYNTHETIC DATA GENERATOR")
    print("=" * 75)

    original_df = read_original_arff(ORIGINAL_FILE)

    # Remove PO for the 11-category experiment.
    original_df = original_df[
        original_df["class"].isin(VALID_CATEGORIES)
    ].reset_index(drop=True)

    print(f"\nOriginal usable PROMISE records: {len(original_df)}")
    print("\nOriginal class distribution:")
    print(original_df["class"].value_counts().sort_index())

    # The fixed split is created BEFORE any synthetic data is generated.
    original_train, original_test = create_fixed_split(original_df)

    print("\nRequested synthetic counts:")
    print(pd.Series(QUOTAS).sort_index())
    print(f"Total synthetic requirements: {sum(QUOTAS.values())}")

    if args.dry_run:
        print("\nDRY RUN: prompts only; no API call made.")
        for category, count in QUOTAS.items():
            if count:
                print("\n" + "=" * 75)
                print(f"{category}: {count}")
                print("=" * 75)
                print(build_prompt(category, count))
        return

    client = get_client()

    generated_rows = []
    requirement_id = 1

    for category, count in QUOTAS.items():
        if count == 0:
            continue

        requirements = generate_category(
            client,
            category,
            count,
        )

        for requirement in requirements:
            generated_rows.append({
                "RequirementID": f"AI2_{requirement_id:03d}",
                "RequirementText": requirement,
                "class": category,
                "Source": "AI_Synthetic",
                "Iteration": "Iteration2",
            })
            requirement_id += 1

    generated_df = pd.DataFrame(generated_rows)

    # Validate against the COMPLETE original dataset, not only the training set.
    validate_generated(generated_df, original_df)

    # Save PROMISE-compatible dataset requested by the project.
    generated_df[
        ["RequirementText", "class"]
    ].to_csv(
        GENERATED_FILE,
        index=False,
        encoding="utf-8",
    )

    # Save metadata separately for research traceability.
    generated_df.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(f"\nSaved generated dataset: {GENERATED_FILE}")
    print(f"Saved audit dataset    : {AUDIT_FILE}")

    # Build the actual Iteration 2 training set.
    build_iteration2_train(
        original_train,
        generated_df,
    )

    print("\n" + "=" * 75)
    print("ITERATION 2 DATASET PREPARATION COMPLETED")
    print("=" * 75)
    print("IMPORTANT:")
    print("- Synthetic data was added only to TRAINING data.")
    print("- The original test set remains untouched.")
    print("- The synthetic dataset is an independent experiment.")
    print("- It is NOT the original paper's Iteration 2 dataset.")


if __name__ == "__main__":
    main()
