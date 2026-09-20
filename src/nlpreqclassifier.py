import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

def preprocess_text(text):
    """Normalizes, tokenizes, removes noise/stopwords, and lemmatizes text."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed)

def load_arff_raw(filepath):
    """Parses ARFF file directly past the @DATA header."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().upper().startswith('@DATA'):
            data_start = i + 1
            break
            
    df = pd.read_csv(filepath, 
                     skiprows=data_start, 
                     header=None, 
                     names=['ProjectID', 'RequirementText', 'class'],
                     quotechar="'", 
                     skipinitialspace=True,
                     on_bad_lines='skip')
    
    # Clean up column names and strip trailing whitespaces
    df['Requirement'] = df['RequirementText'].astype(str).str.strip()
    df['Actual_Category'] = df['class'].astype(str).str.strip()
    
    # Filter out unsupported classes to retain the 11 target categories[cite: 1]
    valid_classes = ['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT']
    df = df[df['Actual_Category'].isin(valid_classes)].copy()
    return df

# 1. Load Baseline (624 items) and Extended Dataset
df_original = load_arff_raw('dataset/original/nfr.arff')
df_extended = load_arff_raw('dataset/expansion/Promise+.arff')

# Clean both sets for duplicate checking
df_original['Clean_Requirement'] = df_original['Requirement'].apply(preprocess_text)
df_extended['Clean_Requirement'] = df_extended['Requirement'].apply(preprocess_text)

# 2. Deduplicate: Remove requirements that exist in original baseline
existing_texts = set(df_original['Clean_Requirement'])
df_candidates = df_extended[~df_extended['Clean_Requirement'].isin(existing_texts)].drop_duplicates(subset=['Clean_Requirement'])

# 3. Sample 113 Requirements for Iteration 2 (Prioritizing minority classes)
needed_count = 202
# Prioritize rare classes first to enhance minority representation[cite: 1]
rare_classes = ['FT', 'L', 'SC', 'MN', 'A', 'LF', 'PE', 'SE', 'US', 'O', 'F']
sampled_frames = []
remaining_needed = needed_count

for cat in rare_classes:
    cat_pool = df_candidates[df_candidates['Actual_Category'] == cat]
    take_n = min(len(cat_pool), int(np.ceil(needed_count / len(rare_classes))))
    if take_n > 0 and remaining_needed > 0:
        take_n = min(take_n, remaining_needed)
        sampled_frames.append(cat_pool.head(take_n))
        remaining_needed -= take_n

df_iteration2_added = pd.concat(sampled_frames)

# Fill any remaining slots if quota not met
if len(df_iteration2_added) < needed_count:
    unselected = df_candidates[~df_candidates.index.isin(df_iteration2_added.index)]
    additional = unselected.head(needed_count - len(df_iteration2_added))
    df_iteration2_added = pd.concat([df_iteration2_added, additional])

# 4. Form the 737-requirement Iteration 2 Dataset[cite: 1]
df_iter2 = pd.concat([df_original, df_iteration2_added]).reset_index(drop=True)
print(f"Iteration 2 Dataset Size: {len(df_iter2)} requirements (Original: {len(df_original)}, Added: {len(df_iteration2_added)})")

# 5. Retrain TF-IDF Vectorizer on Enriched 737 Dataset[cite: 1]
category_profiles = df_iter2.groupby('Actual_Category')['Clean_Requirement'].apply(lambda x: ' '.join(x)).reset_index()
category_names = category_profiles['Actual_Category'].tolist()

vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.95, norm='l2')
category_tfidf = vectorizer.fit_transform(category_profiles['Clean_Requirement'])

# 6. Classification & Validation
def classify_requirement(text):
    req_vector = vectorizer.transform([text])
    similarities = cosine_similarity(req_vector, category_tfidf)[0]
    
    assigned = []
    for i, score in enumerate(similarities):
        cat = category_names[i]
        threshold = 0.75 if cat in ['L', 'FT'] else 0.80
        if score >= threshold:
            assigned.append((cat, score))
            
    if not assigned:
        max_idx = np.argmax(similarities)
        assigned.append((category_names[max_idx], similarities[max_idx]))
        
    assigned.sort(key=lambda x: x[1], reverse=True)
    return assigned[0][0]

df_iter2['Predicted_Category'] = df_iter2['Clean_Requirement'].apply(classify_requirement)

# 7. Output Iteration 2 Performance Metrics
print("\n--- Iteration 2 Performance Metrics (737 Requirements) ---")
print(classification_report(df_iter2['Actual_Category'], df_iter2['Predicted_Category'], target_names=category_names))
# Export Metrics to CSV for the Mini-Project Report
report = classification_report(df_iter2['Actual_Category'], df_iter2['Predicted_Category'], target_names=category_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('iteration3_results.csv')
print("Results exported to iteration3_results.csv")