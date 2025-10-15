from flask import Flask, request, jsonify, send_from_directory
import os
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__, static_folder='web', static_url_path='')

# Load model and vectorizer at startup
MODEL_PATH = 'fake_news_model.pkl'
VECT_PATH = 'vectorizer.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
    model = pickle.load(open(MODEL_PATH, 'rb'))
    vectorizer = pickle.load(open(VECT_PATH, 'rb'))
else:
    model = None
    vectorizer = None

# preprocessing helper
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess(text: str) -> str:
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = [w for w in text.split() if w and w not in stop_words]
    words = [ps.stem(w) for w in words]
    return ' '.join(words)

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/results')
def results_page():
    return send_from_directory('web', 'results.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not found. Run training script first.'}), 500
    data = request.get_json() or {}
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    processed = preprocess(text)
    X = vectorizer.transform([processed]).toarray()
    pred = int(model.predict(X)[0])
    return jsonify({'prediction': 'FAKE' if pred == 1 else 'REAL', 'label': pred})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run a quick model load + predict test')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()

    if args.test:
        if model is None:
            print('Model or vectorizer not found. Please run training script first to create fake_news_model.pkl and vectorizer.pkl')
        else:
            sample = 'Celebrity caught in shocking scandal revealed by insiders'
            print('Sample:', sample)
            print('Prediction:', (model.predict(vectorizer.transform([preprocess(sample)]).toarray())[0]))
    else:
        app.run(host=args.host, port=args.port, debug=True)
