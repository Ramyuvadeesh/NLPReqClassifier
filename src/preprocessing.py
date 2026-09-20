import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
stop_words.discard("shall")   # Preserve 'shall' to match the paper
lemmatizer = WordNetLemmatizer()

def convert_to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    filtered_words = []

    for word in tokens:
        if word not in stop_words:
            filtered_words.append(word)

    return filtered_words

def lemmatize_words(tokens):
    lemmatized_words = []

    for word in tokens:
        lemma = lemmatizer.lemmatize(word)
        lemmatized_words.append(lemma)

    return lemmatized_words

def clean_text(text):
    text = convert_to_lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_words(tokens)

    return " ".join(tokens)

