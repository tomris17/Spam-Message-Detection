# Spam Email / Message Detection Project

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

This repository contains a Natural Language Processing (NLP) classification project built to detect and filter out spam messages and emails[cite: 10].

---

## Dataset Notice
*Note: The dataset used in this project (`spam.csv`)[cite: 10] can be downloaded from standard NLP benchmark repositories such as Kaggle (SMS Spam Collection Dataset).*

---

## Dataset Features
The raw dataset consists of message labels and text contents[cite: 10]:
* **label**: Categorical indicator (`ham` or `spam`)[cite: 10].
* **text**: The raw body text of the message or email[cite: 10].
* **label_num**: Encoded numeric target variable (0 for ham, 1 for spam)[cite: 10].

---

## Project Workflow
1. **Data Preparation**: Selecting relevant columns (`v1`, `v2`), renaming them to `label` and `text`, and mapping categorical values to numeric labels[cite: 10].
2. **Text Vectorization**: Converting textual data into numerical feature matrices using `TfidfVectorizer` with English stop words removal[cite: 10].
3. **Model Training**: Training a `MultinomialNB` classifier on the vectorized text corpus[cite: 10].
4. **Evaluation**: Assessing predictive capability using accuracy score (~96.68%)[cite: 10].
5. **Model Persistence**: Saving the trained estimator and vectorizer using `joblib` into `spam_model.pkl` and `spam_vectorizer.pkl`[cite: 10].
6. **Web Application**: Interactive deployment interface built with Streamlit.

---

## Getting Started & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/spam-message-detection.git](https://github.com/YOUR_USERNAME/spam-message-detection.git)
   cd spam-message-detection
