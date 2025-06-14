# 📰 Fake-News-Detector-TFIDF-PAC

A machine learning project that detects and classifies fake news headlines using TF-IDF vectorization and PassiveAggressiveClassifier.

---

## 📌 Project Overview

This project applies classic NLP and machine learning techniques to differentiate between **REAL** and **FAKE** news. It is built using:

- `TfidfVectorizer` for feature extraction
- `PassiveAggressiveClassifier` for fast and efficient text classification
- `pickle` for saving and loading the trained model and vectorizer

Fake news detection is critical in today’s world of fast-spreading misinformation. This project helps identify unreliable content using labeled news data.

---

## 📂 Dataset

The model is trained on a dataset in `news.csv` with the following structure:
- `text`: News headlines or article snippets
- `label`: Either `"REAL"` or `"FAKE"`

---

🛠️ Requirements

Install the required libraries using pip:


pip install pandas scikit-learn numpy
📈 Model Pipeline
Load and clean the dataset

Encode labels (REAL = 0, FAKE = 1)

Split into training and testing sets

Extract features using TF-IDF

Train PassiveAggressiveClassifier

Evaluate with accuracy and confusion matrix

Save the model and vectorizer with pickle

💻 How to Use
🔁 Train the Model
Run the Jupyter notebook Fake_News_Retrain.ipynb to:

Train the model

Save files as Fake_NEWS_model11.pkl and tf_idf11.pkl


📊 Example Output

Text: NASA Confirms Earth Will Go Completely Dark for 6 Days Starting November 15
Prediction: FAKE

🙋‍♂️ Author
Aryan Patel
