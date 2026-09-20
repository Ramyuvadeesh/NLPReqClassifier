# NLPReqClassifier

## Software Requirement Classification Using NLP, TF-IDF and Cosine Similarity

NLPReqClassifier is a Natural Language Processing (NLP) based system for automatically classifying software requirements into different requirement categories.

The project implements a lightweight similarity-based classification approach using:

- Natural Language Processing (NLP)
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Cosine Similarity
- PROMISE Software Requirements Dataset

The methodology is based on the research paper:

> "An Efficient Methodology for the Categorization of Software Requirements Using Natural Language Processing and Similarity Analysis"

---

## 📌 Project Overview

Software requirements are commonly written in natural language and may belong to different functional and non-functional requirement categories.

Manually classifying a large number of requirements can be time-consuming and error-prone.

This project aims to automate the classification process by:

1. Preprocessing software requirements using NLP techniques.
2. Building category profiles from the training requirements.
3. Converting requirements into TF-IDF vectors.
4. Measuring similarity using Cosine Similarity.
5. Assigning a requirement to the most relevant category.

---

## 🎯 Objectives

The main objectives of this project are:

- To preprocess software requirements using NLP techniques.
- To represent requirements using TF-IDF.
- To create category-based requirement profiles.
- To calculate similarity between a new requirement and category profiles.
- To classify requirements automatically.
- To evaluate the classification performance using standard metrics.
- To reproduce the methodology described in the reference IEEE research paper.
- To extend the system through iterative dataset enhancement.

---

## 🧠 Methodology

The overall workflow of the project is:

```text
Software Requirements
        ↓
Text Preprocessing
        ↓
Tokenization
        ↓
Stopword Removal
        ↓
Lemmatization
        ↓
Category Profile Creation
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Threshold / Fallback Decision
        ↓
Requirement Classification
        ↓
Performance Evaluation
