# Fake News Detection

This project trains simple NLP models (Multinomial Naive Bayes and Logistic Regression) to detect fake news headlines.

Files created by the project:
- `fake_news_model.pkl` — the trained best model
- `vectorizer.pkl` — the CountVectorizer used for preprocessing
- `results.txt` — plaintext summary of metrics
- `web/results.html` — generated HTML report with charts
- `web/index.html`, `web/styles.css`, `web/script.js` — lightweight dashboard UI

Quick start
1. Create a virtual environment (recommended):

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Install requirements:

```cmd
pip install -r requirements.txt
```

3. Train the model (or re-run training and generate reports):

```cmd
python fake_news_detection.py
```

4. Run the server (serves the dashboard and provides `/predict` API):

```cmd
python server.py
```

5. Open the dashboard in your browser:

- http://127.0.0.1:5000/
- http://127.0.0.1:5000/results


https://github.com/user-attachments/assets/2183cfe1-eeaa-4857-a522-319b1767cc67


API
- POST /predict
  - JSON body: {"text": "Your headline here"}
  - Response: {"prediction": "FAKE" | "REAL", "label": 0|1}

Notes
- The dashboard `web/index.html` uses a lightweight client-side demo when no backend is present. When `server.py` is running, the Predict button calls `/predict`.
- If NLTK stopwords are missing, run a small Python one-liner to download them:

```cmd
python -c "import nltk; nltk.download('stopwords')"
```
