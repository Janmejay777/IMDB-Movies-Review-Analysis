
🎥 IMDB Movie Reviews Sentiment Analysis
A Natural Language Processing (NLP) project that performs sentiment analysis on 50,000 IMDB movie reviews, classifying them as positive or negative. This project demonstrates the implementation of a complete NLP pipeline, including data preprocessing, text vectorization, model training, and evaluation.

📌 Project Overview
This project focuses on applying various NLP techniques and machine learning algorithms to classify movie reviews based on sentiment. It uses multiple text vectorization methods and an ensemble learning approach to improve accuracy in sentiment classification.

🛠️ Features
📊 50,000 IMDB movie reviews analyzed

🔍 Sentiment Analysis: Predicts whether a review is positive or negative

🧵 Complete NLP Pipeline from text preprocessing to model evaluation

📈 Accuracy optimization using hyperparameter tuning and advanced vectorization methods

🌲 Random Forest Classifier employed as the primary machine learning model

📝 Comparison of different text vectorization techniques

📚 Technologies & Tools
Python

NLTK

scikit-learn

Gensim

Pandas

NumPy

Matplotlib / Seaborn

📑 Workflow
1️⃣ Data Preprocessing
Removal of HTML tags, special characters, stopwords

Conversion to lowercase

Tokenization and lemmatization

2️⃣ Text Vectorization
Implemented and compared three different text vectorization techniques:

Bag of Words (BoW)

TF-IDF (Term Frequency-Inverse Document Frequency)

Word2Vec (using Gensim)

3️⃣ Model Building
Random Forest Classifier selected as the primary model

Models trained using each vectorized dataset

Hyperparameter tuning performed to enhance model performance

4️⃣ Model Evaluation
Accuracy, Precision, Recall, and F1-Score used for evaluation

Confusion Matrix and classification reports generated for analysis

📊 Results
Notable improvement in accuracy after text preprocessing and hyperparameter tuning

Random Forest with TF-IDF and Word2Vec embeddings produced the best results

