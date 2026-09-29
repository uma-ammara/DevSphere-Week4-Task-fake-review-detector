#  Fake Product Review Detection Using a Neural Network

A beginner-friendly machine learning project that detects whether a product/hotel review is **Likely Genuine** or **Potentially Fake**, using **TF-IDF** text features and a simple **feed-forward neural network** built with **TensorFlow/Keras**. Includes an interactive **Streamlit** web app for live predictions.

---

## 📌 Project Overview

Fake reviews mislead customers and damage trust in online platforms. This project builds a lightweight, explainable neural network that classifies a review as **Genuine** or **Fake** based on its text.

- **Type:** Binary text classification
- **Model:** Simple feed-forward (Dense) neural network — no pretrained models, no complex architectures
- **Feature extraction:** TF-IDF (unigrams + bigrams)
- **Interface:** Streamlit web app

---

## 📊 Dataset

- **Source:** Kaggle (hotel/product review dataset)
- **Size:** 2,400 reviews
- **Columns:**
  - `text` — the review content
  - `label` — original 3-class label: `genuine`, `fake_ai`, `fake_human`

For this project, `fake_ai` and `fake_human` were merged into a single **Fake** class, giving a binary target:

| Class | Count |
|---|---|
| Genuine (0) | 796 |
| Fake (1) | 1,600 |

After removing duplicates and conflicting labels: **2,396 clean rows** used for training/testing.

---

## 🧹 Preprocessing

- Lowercased all text
- Removed URLs and punctuation (kept letters, numbers, spaces)
- Removed missing values and duplicate rows
- Removed reviews with conflicting labels (same text, different label)
- Converted labels to binary (0 = Genuine, 1 = Fake)

---

## 🔤 Feature Extraction — TF-IDF

- `max_features=5000`
- `ngram_range=(1,2)` — unigrams and bigrams
- `min_df=2`
- Fit **only on training data** to avoid data leakage

---

## 🧠 Model Architecture

```
Input (5000 features)
   ↓
Dense(64, activation="relu")
Dropout(0.5)
   ↓
Dense(32, activation="relu")
Dropout(0.3)
   ↓
Dense(1, activation="sigmoid")
```

- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Class weighting:** applied to handle the Genuine/Fake imbalance
- **Early stopping:** on validation loss, patience = 5, restores best weights

---

## 📈 Training Results

Training stopped automatically at **epoch 9** (best weights from **epoch 4**, lowest validation loss = 0.1916).

| Metric | Training | Validation |
|---|---|---|
| Accuracy | ~99.9% | ~91.4% (best) |
| Loss | ~0.01 | ~0.19 (best) |

*(Training accuracy is much higher than validation accuracy — a normal, mild overfitting pattern for a small dataset, controlled with Dropout and Early Stopping.)*

---

## ✅ Test Set Performance (480 unseen reviews)

| Metric | Score |
|---|---|
| **Accuracy** | 90.6% |
| **Precision** | 95.1% |
| **Recall** | 90.7% |
| **F1-score** | 92.8% |

**Classification report:**

```
              precision    recall  f1-score   support

     Genuine       0.83      0.91      0.86       159
        Fake       0.95      0.91      0.93       321

    accuracy                           0.91       480
   macro avg       0.89      0.91      0.90       480
weighted avg       0.91      0.91      0.91       480
```

---

## 🔍 Example Predictions on New Reviews

| Review (excerpt) | Prediction | P(Fake) |
|---|---|---|
| "We stayed here for three nights. The room was clean and the staff helped us find a good restaurant nearby..." | Likely Genuine | 0.076 |
| "Exceptional accommodation. Superior service. Outstanding facilities. Highly recommended..." | Potentially Fake | 0.980 |
| "The breakfast was okay, nothing special. Our shower drained slowly..." | Likely Genuine | 0.052 |
| "This hotel offers a comprehensive range of amenities. The location is convenient..." | Potentially Fake | 0.972 |
| "Loved it! Best hotel ever, everything was perfect, you must book now, five stars!!!" | Potentially Fake | 0.879 |

The model correctly separates natural, detail-specific reviews from generic, overly polished, or overly promotional ones.

---

## 📁 Repository Structure

```
fake-review-detector/
├── notebook/
│   └── Fake_Review_Detection.ipynb    # Full training notebook (Colab)
├── app/
│   ├── app.py                         # Streamlit web app
│   ├── fake_review_model.keras        # Saved trained model
│   └── tfidf_vectorizer.joblib        # Saved TF-IDF vectorizer
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fake-review-detector.git
cd fake-review-detector/app
```

### 2. Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🖥️ App Features

- Paste any product/hotel review into the text box
- Click **Analyze Review**
- Get:
  - Prediction: **Likely Genuine** or **Potentially Fake**
  - Probability breakdown (Genuine % vs Fake %)
  - Visual confidence indicator

---

## ⚠️ Limitations

- Trained mainly on hotel-style reviews — accuracy may drop on other product categories
- TF-IDF captures word patterns, not deep meaning or word order
- Genuine-class precision (83%) is lower than Fake-class precision (95%), due to class imbalance in training data
- Detects writing **style**, not factual truth — cannot verify if a reviewer actually used the product


---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (TF-IDF, train/test split, metrics)
- TensorFlow / Keras
- Matplotlib, Seaborn
- Streamlit

---

