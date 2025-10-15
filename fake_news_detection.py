# ===============================================
# Fake News Detection using NLP and ML
# Author: Siva
# ===============================================

# ---------- Import Libraries ----------
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle
# removed unused plotting and random imports to speed up and avoid unused warnings

# ---------- Download NLTK Resources ----------
# Download stopwords once (quiet) if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ---------- Load Dataset ----------
print("Loading dataset (will sample for speed)...")
df = pd.read_csv('kaggle_fake_train.csv')  # Make sure this file is in the same directory
# To make the script fast for quick runs, sample a subset if the dataset is large
if len(df) > 5000:
    df = df.sample(n=5000, random_state=0).reset_index(drop=True)
print(f"Dataset loaded successfully with shape: {df.shape}")

# ---------- Data Cleaning ----------
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
print("Dropped 'id' column.")

# Remove rows with missing values
df.dropna(inplace=True)

# Reset index to fix KeyError
df.reset_index(drop=True, inplace=True)
print(f"After dropping NaN values and resetting index: {df.shape}")

# ---------- Text Preprocessing ----------
print("Cleaning and preprocessing text (fast)...")
# Cache stopwords set once
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
corpus = []
for title in df['title'].fillna(''):
    s = re.sub('[^a-zA-Z]', ' ', title)
    s = s.lower()
    words = [w for w in s.split() if w and w not in stop_words]
    words = [ps.stem(w) for w in words]
    corpus.append(' '.join(words))

print("Text preprocessing complete!")

# ---------- Bag of Words Model ----------
print("Creating Bag of Words model (smaller, faster)...")
# Use fewer features and only unigrams+bigrams for speed
cv = CountVectorizer(max_features=1000, ngram_range=(1,1))
X = cv.fit_transform(corpus).toarray()
y = df['label']

print(f"Feature matrix shape: {X.shape}")

# ---------- Split Dataset ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ---------- Multinomial Naive Bayes ----------
print("\nTraining Multinomial Naive Bayes model (fast)...")
nb_classifier = MultinomialNB(alpha=0.3)
nb_classifier.fit(X_train, y_train)
nb_y_pred = nb_classifier.predict(X_test)

nb_acc = accuracy_score(y_test, nb_y_pred)
nb_prec = precision_score(y_test, nb_y_pred)
nb_rec = recall_score(y_test, nb_y_pred)
print(f"Naive Bayes -> Accuracy: {nb_acc*100:.2f}%, Precision: {nb_prec:.2f}, Recall: {nb_rec:.2f}")

# ---------- Logistic Regression ----------
print("\nTraining Logistic Regression model (solver=saga, max_iter=300)...")
# Use a faster solver and fewer iterations for a quick run
lr_classifier = LogisticRegression(C=0.8, random_state=0, max_iter=200, solver='liblinear')
lr_classifier.fit(X_train, y_train)
lr_y_pred = lr_classifier.predict(X_test)

lr_acc = accuracy_score(y_test, lr_y_pred)
lr_prec = precision_score(y_test, lr_y_pred)
lr_rec = recall_score(y_test, lr_y_pred)
print(f"Logistic Regression -> Accuracy: {lr_acc*100:.2f}%, Precision: {lr_prec:.2f}, Recall: {lr_rec:.2f}")

# ---------- Confusion Matrices ----------
print("\nConfusion matrices computed (plotting suppressed for fast run).")
nb_cm = confusion_matrix(y_test, nb_y_pred)
lr_cm = confusion_matrix(y_test, lr_y_pred)
print("Naive Bayes confusion matrix:\n", nb_cm)
print("Logistic Regression confusion matrix:\n", lr_cm)

# ---------- Save the Best Model ----------
best_model = lr_classifier if lr_acc > nb_acc else nb_classifier
model_name = "Logistic Regression" if lr_acc > nb_acc else "Multinomial Naive Bayes"
print(f"\nSaving the best model: {model_name}")
pickle.dump(best_model, open('fake_news_model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))

# ---------- Save results summary to a file ----------
results_text = []
results_text.append('Multinomial Naive Bayes results:')
results_text.append(f'Accuracy: {nb_acc*100:.2f}%')
results_text.append(f'Precision: {nb_prec:.2f}')
results_text.append(f'Recall: {nb_rec:.2f}')
results_text.append('Confusion matrix:')
results_text.append(str(nb_cm.tolist()))
results_text.append('')
results_text.append('Logistic Regression results:')
results_text.append(f'Accuracy: {lr_acc*100:.2f}%')
results_text.append(f'Precision: {lr_prec:.2f}')
results_text.append(f'Recall: {lr_rec:.2f}')
results_text.append('Confusion matrix:')
results_text.append(str(lr_cm.tolist()))

with open('results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results_text))

print("Results summary written to results.txt")

# ---------- Generate styled HTML results for the web UI ----------
html_template = """<!doctype html>
<html lang=\"en\"> 
<head>
    <meta charset=\"utf-8\"> 
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"> 
    <title>Fake News Detection - Results</title>
    <link rel=\"stylesheet\" href=\"styles.css\"> 
    <style>
        /* small overrides for the standalone results file */
        body{background:linear-gradient(135deg,#02121a,#08303a);padding:2rem;font-family:Inter,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#e6f0f2}
        .container{max-width:900px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:1rem;align-items:start}
        .panel{background:rgba(255,255,255,0.04);padding:1rem;border-radius:12px;box-shadow:0 10px 30px rgba(2,6,23,0.6);transform-origin:center;animation:float 1s ease both}
        h1{text-align:center;margin-bottom:1rem}
        pre{white-space:pre-wrap;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", "Segoe UI Mono", monospace}
        @keyframes float{0%{transform:translateY(6px);opacity:0}100%{transform:none;opacity:1}}
    </style>
</head>
<body>
    <h1>Fake News Detection — Results</h1>
    <div class=\"container\"> 
        <div class=\"panel\"> 
            <h2>Multinomial Naive Bayes</h2>
            <pre>
Accuracy: %%NB_ACC%%
Precision: %%NB_PREC%%
Recall: %%NB_RECALL%%
Confusion matrix:
[%%NB_CM00%%  %%NB_CM01%%]
[%%NB_CM10%%  %%NB_CM11%%]
            </pre>
        </div>
        <div class=\"panel\"> 
            <h2>Logistic Regression</h2>
            <pre>
Accuracy: %%LR_ACC%%
Precision: %%LR_PREC%%
Recall: %%LR_RECALL%%
Confusion matrix:
[%%LR_CM00%%  %%LR_CM01%%]
[%%LR_CM10%%  %%LR_CM11%%]
            </pre>
        </div>
    </div>
    <p style=\"text-align:center;opacity:.8;margin-top:1rem\">This page was generated by the training script. Open <a href=\"index.html\">dashboard</a> for an interactive preview.</p>
    <script>
        // small fade-in for panels
        document.querySelectorAll('.panel').forEach(function(p,i){p.style.animationDelay=(i*120)+'ms'});
    </script>
</body>
</html>
"""

import os
html_content = html_template.replace('%%NB_ACC%%', f'{nb_acc*100:.2f}%')
html_content = html_content.replace('%%NB_PREC%%', f'{nb_prec:.2f}')
html_content = html_content.replace('%%NB_RECALL%%', f'{nb_rec:.2f}')
html_content = html_content.replace('%%NB_CM00%%', str(nb_cm[0,0]))
html_content = html_content.replace('%%NB_CM01%%', str(nb_cm[0,1]))
html_content = html_content.replace('%%NB_CM10%%', str(nb_cm[1,0]))
html_content = html_content.replace('%%NB_CM11%%', str(nb_cm[1,1]))

html_content = html_content.replace('%%LR_ACC%%', f'{lr_acc*100:.2f}%')
html_content = html_content.replace('%%LR_PREC%%', f'{lr_prec:.2f}')
html_content = html_content.replace('%%LR_RECALL%%', f'{lr_rec:.2f}')
html_content = html_content.replace('%%LR_CM00%%', str(lr_cm[0,0]))
html_content = html_content.replace('%%LR_CM01%%', str(lr_cm[0,1]))
html_content = html_content.replace('%%LR_CM10%%', str(lr_cm[1,0]))
html_content = html_content.replace('%%LR_CM11%%', str(lr_cm[1,1]))

os.makedirs('web', exist_ok=True)
with open(os.path.join('web','results.html'), 'w', encoding='utf-8') as hf:
        hf.write(html_content)

print("Generated web/results.html")

# ---------- Prediction Function ----------
def fake_news(sample_news):
    sample_news = re.sub('[^a-zA-Z]', ' ', sample_news)
    sample_news = sample_news.lower()
    sample_news_words = sample_news.split()
    sample_news_words = [word for word in sample_news_words if word not in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_news = [ps.stem(word) for word in sample_news_words]
    final_news = ' '.join(final_news)
    temp = cv.transform([final_news]).toarray()
    return best_model.predict(temp)[0]

# ---------- Quick Self-test with a few hardcoded samples (no external test CSV) ----------
print("\nTesting model with a few sample inputs...")
sample_titles = [
    "President signs new bill to improve healthcare",
    "Celebrity caught in shocking scandal revealed by insiders",
    "Scientists discover water on Mars in unprecedented quantities"
]
for sample_news in sample_titles:
    prediction = fake_news(sample_news)
    print("\nNews:", sample_news)
    print("Prediction:", "FAKE" if prediction == 1 else "REAL")