"""
live_demo.py

============================================================
NLPReqClassifier - LIVE INTERACTIVE PREDICTION
============================================================
Trains on the complete 984-record master dataset (Original + 
Synthetic Iterations) and opens an interactive prompt for 
real-time requirement classification.
"""

import os
import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIGURATION
# ============================================================

GENERAL_THRESHOLD = 0.80
UNDERREPRESENTED_THRESHOLD = 0.75
UNDERREPRESENTED_CATEGORIES = {"L", "FT"}

# ============================================================
# NLTK SETUP & PREPROCESSING
# ============================================================

def initialize_nltk():
    print("Initializing NLP resources...")
    resources = [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab")
    ]
    for resource_name, resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    processed_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]
    return " ".join(processed_tokens)

# ============================================================
# DATASET LOADING
# ============================================================

def load_promise_dataset(file_path):
    if not os.path.exists(file_path):
        return None

    data_start = None
    with open(file_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if line.strip().upper() == "@DATA":
                data_start = i + 1
                break

    if data_start is None:
        return None

    df = pd.read_csv(
        file_path,
        skiprows=data_start,
        header=None,
        names=["ProjectID", "RequirementText", "class"],
        quotechar="'",
        skipinitialspace=True
    )

    df = df.dropna(subset=["RequirementText", "class"]).copy()
    df["class"] = df["class"].astype(str).str.strip()
    df = df[df["class"] != "PO"].copy()
    return df.reset_index(drop=True)

def preprocess_dataset(df):
    df["Clean_Requirement"] = df["RequirementText"].apply(preprocess_text)
    return df

# ============================================================
# MODEL TRAINING
# ============================================================

def build_category_profiles(df):
    category_profiles = df.groupby("class")["Clean_Requirement"].apply(lambda values: " ".join(values)).reset_index()
    category_profiles.columns = ["Category", "CategoryDocument"]
    return category_profiles.sort_values("Category").reset_index(drop=True)

def train_tfidf(category_profiles):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=0.01,
        max_features=2000,
        max_df=0.95,
        norm="l2"
    )
    documents = category_profiles["CategoryDocument"].tolist()
    category_tfidf = vectorizer.fit_transform(documents)
    category_names = category_profiles["Category"].tolist()
    return vectorizer, category_tfidf, category_names

# ============================================================
# PREDICTION LOGIC
# ============================================================

def classify_requirement(cleaned_text, vectorizer, category_tfidf, category_names):
    requirement_vector = vectorizer.transform([cleaned_text])
    similarities = cosine_similarity(requirement_vector, category_tfidf)[0]

    scores = []
    for i, category in enumerate(category_names):
        score = float(similarities[i])
        threshold = UNDERREPRESENTED_THRESHOLD if category in UNDERREPRESENTED_CATEGORIES else GENERAL_THRESHOLD
        scores.append({
            "Category": category,
            "Similarity": score,
            "Threshold": threshold,
            "ReachedThreshold": score >= threshold
        })

    scores.sort(key=lambda item: item["Similarity"], reverse=True)
    selected = [item for item in scores if item["ReachedThreshold"]]
    
    used_fallback = False
    if len(selected) == 0:
        selected = [scores[0]]
        used_fallback = True
    else:
        selected.sort(key=lambda item: item["Similarity"], reverse=True)

    return selected[0]["Category"], scores, used_fallback

# ============================================================
# INTERACTIVE DEMO
# ============================================================

def interactive_testing(vectorizer, category_tfidf, category_names):
    print("\n" + "=" * 70)
    print("LIVE PREDICTION MODE (TRAINED ON MASTER DATASET)")
    print("=" * 70)
    print("Type your software requirement below.")
    print("Type 'exit' or 'quit' to end the program.\n")

    while True:
        user_input = input("Enter a requirement: ")
        
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Exiting live prediction mode...")
            break
            
        if not user_input.strip():
            continue

        cleaned_text = preprocess_text(user_input)
        predicted, scores, used_fallback = classify_requirement(cleaned_text, vectorizer, category_tfidf, category_names)
        top_similarity = scores[0]["Similarity"]
        
        print(f"\n--> Predicted Category : {predicted}")
        print(f"--> Similarity Score   : {top_similarity:.4f}")
        
        if used_fallback:
            print("--> Note: Score fell below strict threshold; used fallback maximum similarity.")
        print("-" * 50)

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("INITIALIZING FULL MODEL FOR LIVE DEMO")
    print("=" * 70)

    initialize_nltk()

    print("\nLoading and combining all datasets into Master Training Set...")
    
    files_to_load = [
        "dataset/original/nfr.arff",
        "dataset/original/synthetic_test_113.arff",
        "dataset/original/synthetic_test_89.arff",
        "dataset/original/synthetic_test_81.arff",
        "dataset/original/synthetic_test_77_erp.arff"
    ]
    
    dataframes = []
    for file_path in files_to_load:
        df = load_promise_dataset(file_path)
        if df is not None:
            df = preprocess_dataset(df)
            dataframes.append(df)
        else:
            print(f"Warning: Could not find {file_path}. Skipping.")
            
    if not dataframes:
        print("Error: No datasets found. Please ensure your ARFF files are in the 'dataset' folder.")
        return
        
    df_master = pd.concat(dataframes, ignore_index=True)
    print(f"Master Training Set Built: {len(df_master)} records")

    print("\nBuilding Category Profiles and Training TF-IDF Vectorizer...")
    category_profiles = build_category_profiles(df_master)
    vectorizer, category_tfidf, category_names = train_tfidf(category_profiles)

    interactive_testing(vectorizer, category_tfidf, category_names)

if __name__ == "__main__":
    main()