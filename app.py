import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template
from typing import cast  # used for silencing type checker if needed

# Load model
with open("Fake_NEWS_model11.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("tf_idf11.pkl", "rb") as f:
    tfidf = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home() -> str:  # Flask expects a string (HTML) response here
    return cast(str, render_template('index.html'))

@app.route('/predict', methods=['POST'])
def predict() -> str:
    if request.method == 'POST':
        news = request.form['news']
        data = [news]
        vect = tfidf.transform(data)
        prediction = model.predict(vect)
        result = "FAKE NEWS" if prediction[0] == 1 else "REAL NEWS"
        return cast(str, render_template('index.html', prediction_text=f'Prediction: {result}'))
    return cast(str, render_template('index.html', prediction_text=''))  # fallback

if __name__ == '__main__':
    app.run(debug=True)
