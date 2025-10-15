# Fake News Detection Project

This project trains simple NLP models (Multinomial Naive Bayes and Logistic Regression) to detect fake news headlines.

---

## Project Structure

```
FAKE_NEWS_PROJECT/
│
├── .venv/                       # Python virtual environment
├── web/
│   ├── index.html               # Dashboard UI
│   ├── results.html             # Generated HTML report with charts
│   ├── script.js                # Dashboard JS
│   ├── styles.css               # Dashboard CSS
│   └── logo.jpg                 # Project logo
├── fake_news_detection.py       # Script to train models and generate results
├── server.py                    # Flask server for API & dashboard
├── fake_news_model.pkl          # Trained model
├── vectorizer.pkl               # CountVectorizer for preprocessing
├── results.txt                  # Summary of metrics
├── kaggle_fake_train.csv        # Training dataset
├── kaggle_fake_test.csv         # Test dataset
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## Quick Start

### 1. Set up virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# OR
source .venv/bin/activate  # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model and generate reports

```bash
python fake_news_detection.py
```

This will generate:

* `fake_news_model.pkl` — trained best model
* `vectorizer.pkl` — CountVectorizer used for preprocessing
* `results.txt` — text summary of metrics
* `web/results.html` — HTML report with charts

### 4. Run the dashboard server

```bash
python server.py
```

Open the dashboard in your browser:

* Dashboard: http://127.0.0.1:5000/
* Results Report: http://127.0.0.1:5000/results

---

## Demo Output

**Example Predictions:**

| News Headline                                                 | Prediction |
| ------------------------------------------------------------- | ---------- |
| President signs new bill to improve healthcare                | REAL       |
| Celebrity caught in shocking scandal revealed by insiders     | FAKE       |
| Scientists discover water on Mars in unprecedented quantities | FAKE       |

**Demo Video:**

https://github.com/user-attachments/assets/e39586f5-69db-4e1b-be53-5def0a34ba20

---

## API Usage

**POST /predict**

* Request body (JSON):

```json
{"text": "Your headline here"}
```

* Response:

```json
{"prediction": "FAKE" | "REAL", "label": 0|1}
```

---

## Notes

* The dashboard `web/index.html` has a lightweight client-side demo when no backend is running. When `server.py` is running, the Predict button calls `/predict`.
* If NLTK stopwords are missing, download them:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

