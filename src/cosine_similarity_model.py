import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from load_data import load_dataset
from preprocessing import clean_text

df = load_dataset()

df["Cleaned_Text"] = df["RequirementText"].apply(clean_text)

vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=0.01,
    max_features=2000,
    ngram_range=(1, 2),
    norm="l2"
)

tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Text"])

similarity_matrix = cosine_similarity(tfidf_matrix)

print("Similarity Matrix Shape:")
print(similarity_matrix.shape)

similarity_matrix_no_self = similarity_matrix.copy()

np.fill_diagonal(similarity_matrix_no_self, 0)

most_similar_index = similarity_matrix_no_self[0].argmax()

print("\nMost Similar Requirement Index:", most_similar_index)

print("Similarity Score:",
      similarity_matrix_no_self[0][most_similar_index])

print("\nOriginal Requirement:")
print(df["RequirementText"][0])

print("\nMost Similar Requirement:")
print(df["RequirementText"][most_similar_index])

print("\nOriginal Class:", df["class"][0])

print("Predicted Class:", df["class"][most_similar_index])