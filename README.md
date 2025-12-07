# 🎵 Sentiment Analysis on Pop Music Reddit Posts

### NLP Project using a 32K Short-Form Reddit Sentiment Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NLP](https://img.shields.io/badge/Task-Text%20Classification-orange)
![Models](https://img.shields.io/badge/Models-BERT%2FLR%2FDF-purple)

## 📌 Overview
This project performs **Sentiment Analysis** on short-form Reddit posts related to pop music, artists, fandom culture, and industry events.

The dataset is fully labeled using a BERT-based transformer model, making it ideal for:
* **NLP Beginners**
* **Machine Learning Practitioners**
* **Deep Learning Model Training**
* **Text Classification Projects**
* **Benchmarking Transformer Models**

**The project covers:**
* ✔ Text preprocessing and cleaning
* ✔ Exploratory text analysis (EDA)
* ✔ Vectorization techniques (TF-IDF, Embeddings)
* ✔ Training both ML & DL (Transformer) models
* ✔ Performance evaluation using standard metrics
* ✔ Extracting insights from social media data

## 🔗 Dataset Source
**Kaggle Dataset:** ![Kaggle dataset](https://www.kaggle.com/datasets/alyahmedts13/reddit-sentiment-analysis-dataset-for-nlp-projects)

---

## 📂 About the Dataset
The dataset contains approximately **32,000 short Reddit posts** ($\le 280$ characters) discussing major pop culture entities and events, including:
* Taylor Swift
* Olivia Rodrigo
* Grammy Awards
* Billboard charts
* Pop artist controversies and releases

The posts were primarily collected from subreddits such as: `r/popheads`, `r/Music`, and `r/Billboard`.

### 📦 Dataset Size
* **Total Posts:** ~32,000 (Cleaned and Filtered)
* **Original Collected Posts:** ~124,000
* **Language:** English
* **Post Types:** Titles + short text bodies combined

### 🏷️ Sentiment Labels
Labels were generated using the **CardiffNLP Twitter RoBERTa** model, fine-tuned for social media sentiment. This provides a robust ground truth for training:

| Label | Meaning |
| :--- | :--- |
| **positive** | Happy, excited, supportive, praising |
| **neutral** | Observational, factual, non-emotional |
| **negative** | Angry, dismissive, critical, toxic |

---

## 🛠 Features of This Project
* ✔ Cleaned and filtered English Reddit text for analysis.
* ✔ Data is preprocessed and ready for NLP workflows.
* ✔ **Balanced** representation of sentiment classes.
* ✔ Suitable for benchmarking text classification experiments.
* ✔ Captures real-world social media sentiment noise.

### 🧹 Data Cleaning Steps
1.  Removed non-English posts.
2.  Removed long posts (limit of 280 characters).
3.  Merged the original title and body text into a single processing column.
4.  Filtered out deleted or empty content.
5.  Cleaned URLs, emojis, and unwanted symbols.
6.  Lowercased the text and stripped extra whitespace.

---

## 🔍 Exploratory Analysis
The included notebook allows for deep exploration of the data:
* **Word Clouds:** Visualization of most frequent words per sentiment class 
* **Label Distribution:** Visualizing the balance of positive, neutral, and negative labels.
* **Artist-wise Trends:** Analyzing sentiment shifts specific to certain pop stars.
* **Discourse Analysis:** Examining the prevalence of slang, emojis, and toxicity patterns.

---

## 🤖 Model Training Workflow

### 1️⃣ Text Preprocessing
* Tokenization
* Stopword removal
* Lemmatization (optional)
* Cleaning URLs, emojis, and user mentions

### 2️⃣ Vectorization Approaches
You can experiment with various vectorization and embedding techniques:
* Traditional: **TF-IDF**, **Bag-of-Words**
* Dense Embeddings: **Word2Vec**, **FastText**
* Transformer Embeddings: **BERT** and related models

### 3️⃣ ML Models
Start with baselines using classical Machine Learning models:
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Naïve Bayes

### 4️⃣ Transformer Models
Move to state-of-the-art Deep Learning models:
* BERT
* DistilBERT
* RoBERTa (and the original **Twitter RoBERTa**)
* Electra

### 5️⃣ Evaluation Metrics
Use these metrics to assess model performance:

| Metric | Use |
| :--- | :--- |
| **Accuracy** | Overall proportion of correct predictions |
| **F1-score** | Harmonic mean of precision and recall (best for imbalanced data) |
| **Confusion Matrix** | Detailed insights into misclassifications (False Positives/Negatives) |
| **ROC-AUC** | Performance metric for binary classification (used with positive/non-positive) |

---
