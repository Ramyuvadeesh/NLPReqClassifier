from sklearn.feature_extraction.text import TfidfVectorizer

from load_data import load_dataset
from preprocessing import clean_text

# Load dataset
df = load_dataset()

# Apply preprocessing to every requirement
df["Cleaned_Text"] = df["RequirementText"].apply(clean_text)

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    norm="l2"
)

# Convert text into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Text"])

print("=" * 50)
print("TF-IDF MATRIX INFORMATION")
print("=" * 50)

print("\nShape of TF-IDF Matrix:")
print(tfidf_matrix.shape)

print("\nVocabulary Size:")
print(len(vectorizer.vocabulary_))

print("\nFirst 20 Features:")
print(vectorizer.get_feature_names_out()[:20])

print("\nFirst Requirement:")
print(df["RequirementText"][0])

print("\nAfter Preprocessing:")
print(df["Cleaned_Text"][0])

print("\nSparse Vector:")
print(tfidf_matrix[0])

