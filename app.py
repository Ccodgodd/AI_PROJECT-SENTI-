# ---------------------------------------------------
# Sentiment Analysis Flask App (GPU-Optimized Version)
# ---------------------------------------------------

# Run this first in terminal:
# pip install flask pandas scikit-learn joblib matplotlib tensorflow wordcloud

from flask import Flask, render_template, request, jsonify
import pandas as pd, re, os, joblib, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from wordcloud import WordCloud
import tensorflow as tf
import os

# ---------------------------------------------------
# GPU / CPU DETECTION
# ---------------------------------------------------
# ---------------------------------------------------
# GPU / CPU DETECTION (Windows-safe version)
# ---------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected! Training will use GPU for acceleration.")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Warning: GPU memory growth setting failed:", e)
else:
    print("No GPU detected — running on CPU (training may be slower).")


# ---------------------------------------------------
# Flask App Setup
# ---------------------------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
RNN_MODEL_PATH = "rnn_model.h5"

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
vectorizer = joblib.load(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else None
data = None

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'rt[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def normalize_label(raw_value):
    s = str(raw_value).strip().lower()
    if s in {"-1", "-1.0"}:
        return -1
    if s in {"0", "0.0", "neutral", "neu"}:
        return 0
    if s in {"1", "1.0"}:
        return 1
    negative_aliases = {"neg", "negative", "bad", "terrible", "awful", "hate", "horrible", "worst"}
    positive_aliases = {"pos", "positive", "good", "great", "excellent", "love", "awesome", "best"}
    neutral_aliases = {"neutral", "neu", "ok", "okay", "fine", "average", "meh", "normal"}
    if s in negative_aliases:
        return -1
    if s in positive_aliases:
        return 1
    if s in neutral_aliases:
        return 0
    return 0

# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global data
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'})
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'CSV error: {str(e)}'})
    preview_html = data.head(10).to_html(classes='styled-table', index=False)
    return jsonify({'preview': preview_html})

@app.route("/preprocess", methods=["POST"])
def preprocess():
    global data
    if data is None:
        return jsonify({'error': 'No dataset found!'})
    text_col = data.columns[0]
    data['clean'] = data[text_col].apply(clean_text)
    preview_html = data[['clean']].head(10).to_html(classes='styled-table', index=False)
    return jsonify({'clean_preview': preview_html})

@app.route("/train_logistic", methods=["POST"])
def train_logistic():
    global data, model, vectorizer
    if data is None:
        return jsonify({'error': 'Upload dataset first!'})

    text_col, label_col = data.columns[0], data.columns[1]
    data['clean'] = data[text_col].apply(clean_text)

    data_sample = data if len(data) <= 400 else data.sample(n=400, random_state=42)

    y = np.array([normalize_label(v) for v in data_sample[label_col]])
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data_sample['clean'], y, test_size=0.1, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, class_weight='balanced')
    model.fit(X_train_vec, y_train)
    log_acc = round(model.score(X_test_vec, y_test) * 100, 2)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    os.makedirs("static/assets", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sentiment_counts = data_sample[label_col].value_counts()
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Sentiment Distribution')
    plt.savefig('static/assets/sentiment_pie.png', dpi=100, bbox_inches='tight')
    plt.close()

    return jsonify({
        'logistic_accuracy': log_acc,
        'sentiment_pie': 'static/assets/sentiment_pie.png',
        'sample_size': len(data_sample)
    })

@app.route("/train_rnn", methods=["POST"])
def train_rnn():
    global data
    if data is None:
        return jsonify({'error': 'Upload dataset first!'})

    text_col, label_col = data.columns[0], data.columns[1]
    data['clean'] = data[text_col].apply(clean_text)

    data_half = data.sample(frac=0.5, random_state=42) if len(data) > 2 else data

    y = np.array([normalize_label(v) for v in data_half[label_col]])
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data_half['clean'], y, test_size=0.2, random_state=42, stratify=y
    )

    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)
    max_len = 80
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    def map_to_index(arr):
        mapping = {-1: 0, 0: 1, 1: 2}
        return np.vectorize(mapping.get)(arr)

    y_train_cat = to_categorical(map_to_index(y_train))
    y_test_cat = to_categorical(map_to_index(y_test))
    vocab_size = len(tokenizer.word_index) + 1

    # GPU-Accelerated LSTM Model
    rnn_model = Sequential([
        Embedding(vocab_size, 64),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(y_train_cat.shape[1], activation='softmax')
    ])
    rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=1, restore_best_weights=True)
    hist = rnn_model.fit(X_train_pad, y_train_cat, epochs=3, batch_size=128, validation_split=0.1,
                         callbacks=[early_stop], verbose=1)

    rnn_loss, rnn_acc = rnn_model.evaluate(X_test_pad, y_test_cat, verbose=0)
    rnn_acc = round(rnn_acc * 100, 2)
    rnn_model.save(RNN_MODEL_PATH)

    os.makedirs("static/assets", exist_ok=True)
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(hist.history['accuracy']) + 1)
    plt.plot(epochs, hist.history['accuracy'], 'b-', label='Train Acc', marker='o')
    plt.plot(epochs, hist.history['val_accuracy'], 'r-', label='Val Acc', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('RNN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/assets/rnn_training_accuracy.png', dpi=100, bbox_inches='tight')
    plt.close()

    return jsonify({
        'rnn_accuracy': rnn_acc,
        'rnn_training_chart': 'static/assets/rnn_training_accuracy.png',
        'sample_size': len(data_half)
    })

@app.route("/predict", methods=["POST"])
def predict():
    global model, vectorizer
    if not model or not vectorizer:
        return jsonify({'error': 'Train the model first!'})
    text = request.form.get("text", "")
    if not text.strip():
        return jsonify({'error': 'No text provided!'})
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = int(model.predict(vec)[0])
    return jsonify({'label': pred})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

