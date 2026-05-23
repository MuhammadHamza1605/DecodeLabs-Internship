
#  DecodeLabs AI Industrial Training Kit — Batch 2026


## 📋 Overview

This repository contains **4 progressive AI projects** built as part of the **DecodeLabs Industrial AI Training Program (Batch 2026)**. Each project advances from foundational rule-based logic to applied machine learning, forming a complete journey from deterministic systems to probabilistic intelligence.

```
Project 1 → Rule-Based Chatbot      (Control Flow & Logic)
Project 2 → Data Classification     (Supervised Learning / KNN)
Project 3 → AI Recommendation       (TF-IDF + Cosine Similarity)
Project 4 → Sentiment Analysis      (NLP + Logistic Regression)
```

All projects follow the **IPO (Input → Process → Output)** architecture and are written in **pure Python**, designed to be understandable, extensible, and portfolio-ready.

---

## 🗂️ Repository Structure

```
decodelabs-ai-training-kit/
│
├── project1_chatbot.py          # Rule-Based AI Chatbot
├── project2_classification.py   # KNN Data Classifier (Iris Dataset)
├── project3_recommendation.py   # Tech Stack Recommender (Content-Based)
├── project4_sentiment.py        # NLP Sentiment Analyser
│
├── README.md                    # This file
└── requirements.txt             # Dependencies
```

---

## ⚙️ Requirements

### Python Version
```
Python 3.8+
```

### Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
```

---

## 🚀 Projects

---

### Project 1 — Rule-Based AI Chatbot 🤖

**File:** `project1_chatbot.py`

#### Goal
Build a simple chatbot that responds to predefined user inputs using pure control-flow logic — no machine learning required.

#### Architecture
```
Raw Input → Sanitize (lower + strip) → Dictionary Lookup O(1) → Response Output
                                              ↑
                                    [Knowledge Base]
                                    30+ intent mappings
```

#### Key Concepts
| Concept | Implementation |
|---|---|
| **Infinite Loop** | `while True:` — chatbot heartbeat |
| **Kill Command** | `exit` / `quit` — breaks the loop cleanly |
| **Sanitization** | `raw.lower().strip()` — normalizes all input |
| **O(1) Lookup** | `dict.get(key, fallback)` — professional vs if-elif ladder |
| **Fallback** | Default response for unknown intents |

#### Run
```bash
python project1_chatbot.py
```

#### Sample Session
```
You: hello
DecoBot: Hello! I'm DecoBot 🤖 — your AI assistant. How can I help you today?

You: what is machine learning
DecoBot: Machine Learning is a subset of AI where systems learn patterns from data...

You: exit
DecoBot: Goodbye! Keep building! 🚀
```

#### What You Learn
- Control flow and decision-making logic
- The IPO model applied to conversation
- Why dictionaries (hash maps) outperform if-elif ladders: **O(1) vs O(n)**
- How rule-based systems form the guardrails of modern AI

---

### Project 2 — Data Classification Using AI 📊

**File:** `project2_classification.py`

#### Goal
Build a supervised learning pipeline that classifies iris flowers into 3 species using the K-Nearest Neighbors algorithm.

#### Architecture
```
Iris Dataset (150 samples)
    ↓
StandardScaler (Mean=0, Variance=1)
    ↓
Train-Test Split (80% / 20%, stratified)
    ↓
K-Tuning Loop (K=1 to 20 → find elbow)
    ↓
KNeighborsClassifier.fit(X_train, y_train)
    ↓
Predictions → Confusion Matrix + F1 Score
```

#### Dataset
```
Samples    : 150 (50 per class — balanced)
Classes    : Setosa | Versicolor | Virginica
Features   : sepal_length, sepal_width, petal_length, petal_width
```

#### Results
```
Accuracy  : 96.67%
F1 Score  : 0.9666 (weighted)
Algorithm : KNN with auto-tuned optimal K
```

#### Run
```bash
python project2_classification.py
```

This script automatically trains the model, prints evaluation metrics, and saves a **4-panel dashboard** (`project2_classification.png`) showing:
- K-tuning elbow curve
- Confusion matrix
- Feature scatter plot (petal dims)
- Per-class F1 scores

#### What You Learn
- The full supervised learning pipeline
- Why feature scaling is critical (StandardScaler)
- The bias-variance trade-off when choosing K
- Why accuracy alone is misleading → use F1 + Confusion Matrix

---

### Project 3 — AI Recommendation Logic 🎯

**File:** `project3_recommendation.py`

#### Goal
Build a content-based recommendation engine that matches a user's skills to the most relevant tech career paths — built from scratch without sklearn's recommendation tools.

#### Architecture: 4-Step Pipeline
```
Step 1 — INGEST   : Capture user skills (minimum 3 inputs)
Step 2 — SCORE    : Compute TF-IDF vectors → Cosine Similarity vs all roles
Step 3 — SORT     : Rank all roles by similarity score (descending)
Step 4 — FILTER   : Return Top-N results (prevents choice overload)
```

#### Algorithm Details

**TF-IDF Vectorization** (built from scratch):
```
TF(term, doc)  = count(term in doc) / total_terms(doc)
IDF(term)      = log(total_docs / docs_containing_term)
Weight         = TF × IDF
```
- **TF** rewards terms that appear frequently in a specific role
- **IDF** penalizes generic terms like "software" or "code" that appear everywhere

**Cosine Similarity:**
```
cos(θ) = (A · B) / (||A|| × ||B||)

Score 1.0 → Perfect match (identical orientation)
Score 0.0 → No match (orthogonal vectors)
```

#### Job Roles Catalog (12 roles)
```
Data Scientist          | Machine Learning Engineer  | Data Engineer
DevOps Engineer         | Backend Developer          | Frontend Developer
Cloud Architect         | AI Research Scientist      | Cybersecurity Engineer
Full Stack Developer    | NLP Engineer               | Database Administrator
```

#### Run
```bash
python project3_recommendation.py
```

#### Sample Output
```
Enter at least 3 skills:
  Skill 1: python
  Skill 2: machine learning
  Skill 3: tensorflow
  Skill 4: deep learning

🎯 TOP 3 RECOMMENDED ROLES:
  #1 Machine Learning Engineer
      [█████████████████░░░░░░░░░░░░░░░░░░] 43.1% match

  #2 Data Scientist
      [██████████████░░░░░░░░░░░░░░░░░░░░░] 36.1% match

  #3 NLP Engineer
      [██████████████░░░░░░░░░░░░░░░░░░░░░] 35.5% match
```

#### What You Learn
- Content-based vs collaborative filtering
- TF-IDF: why it outperforms simple binary (0/1) matching
- Cosine Similarity: why it beats Euclidean distance for text
- The Cold Start Problem and how to bypass it
- Building recommendation logic without external libraries

---

### Project 4 — AI Sentiment Analysis 🧠

**File:** `project4_sentiment.py`

#### Goal
Build a full NLP pipeline that classifies text as **Positive**, **Neutral**, or **Negative** using TF-IDF feature extraction and Logistic Regression.

#### Architecture
```
Raw Text
    ↓
Preprocessing (lowercase → remove URLs/symbols → normalize whitespace)
    ↓
TF-IDF Vectorizer (unigrams + bigrams, 1000 features, sublinear TF)
    ↓
Train-Test Split (80% / 20%, stratified)
    ↓
Logistic Regression (multi-class, lbfgs solver)
    ↓
Predictions + Lexicon Signal (secondary validation)
    ↓
Evaluation (Accuracy + F1 + Confusion Matrix)
```

#### Dataset
```
Corpus     : 90 balanced samples
Classes    : Positive (30) | Neutral (30) | Negative (30)
Domain     : Product reviews, social media, general feedback
```

#### Results
```
Accuracy  : 83.33%
F1 Score  : 0.8364 (weighted)
Algorithm : Logistic Regression + TF-IDF Bigrams
```

#### Run
```bash
python project4_sentiment.py
```

Then type any sentence, or type `demo` to auto-run 5 example sentences:

```
Enter text: demo

📝 This AI course is absolutely fantastic I learned so much!...
😊 POSITIVE  | [████████████████████] 94.2%
   Pos=0.942 | Neu=0.031 | Neg=0.027 | Lexicon=POSITIVE

📝 The product broke after two days complete waste of money...
😞 NEGATIVE  | [██████████████████░░] 89.1%
   Pos=0.023 | Neu=0.086 | Neg=0.891 | Lexicon=NEGATIVE
```

#### What You Learn
- Complete NLP preprocessing pipeline
- TF-IDF with bigrams for richer context capture
- The "accuracy mirage" — why F1 + confusion matrix matters
- Combining ML predictions with lexicon-based signals
- Real-world pattern: model + rule layer = robust system

---

## 🧠 Learning Progression

```
PROJECT 1                PROJECT 2                PROJECT 3                PROJECT 4
Rule-Based Logic    →    Supervised Learning  →   Similarity Math     →   NLP Pipeline
──────────────           ──────────────────       ───────────────          ──────────────
if-else / dict           KNN algorithm            TF-IDF from scratch      TF-IDF + LogReg
while loop               Train/test split          Cosine similarity        Text preprocessing
O(1) lookup              Feature scaling           Content filtering        Multi-class NLP
Control flow             Confusion matrix          Cold start problem       Lexicon signals
IPO model                F1 Score / Recall         Top-N ranking            Bigram features
```

---

## 📈 Results Summary

| Project | Algorithm | Metric | Score |
|---|---|---|---|
| 1 — Chatbot | Rule-Based Dictionary | Intent Coverage | 30+ intents |
| 2 — Classification | KNN (Auto K-Tuning) | Accuracy | **96.67%** |
| 3 — Recommendation | TF-IDF + Cosine Similarity | Top-3 Precision | Content-Based |
| 4 — Sentiment | Logistic Regression + TF-IDF | Accuracy | **83.33%** |

---

## 🏗️ The White Box Principle

All four projects implement **deterministic, interpretable systems** — the opposite of black-box deep learning. This is intentional:

> *"The rule-based generative program is a white box — the interpreter can always provide straightforward explanations."*

**Why this matters:**
- ✅ **Traceability** — Input → Logic → Output. No mystery.
- ✅ **Safety** — Zero hallucination risk in rule-based layers.
- ✅ **Compliance** — Essential for Finance & Healthcare domains.
- ✅ **Foundation** — You must master logic before you can manage probability.

---

## 🔮 What's Next

Having completed these 4 projects, the natural progression is:

```
Project 1-4 (This Kit)          →    Advanced AI
────────────────────────              ──────────────────────────────
Rule-based chatbot              →    LLM-powered conversational AI
KNN classification              →    Deep neural networks
Content-based recommendation    →    Neural collaborative filtering
Bag-of-words NLP                →    Transformers (BERT, GPT)
```

---

## 👥 About DecodeLabs

DecodeLabs is an AI training organization based in Greater Lucknow, India, running hands-on industrial AI training programs for aspiring engineers.

| | |
|---|---|
| 📞 Phone | +91 89330 06408 |
| ✉️ Email | decodelabs.tech@gmail.com |
| 🌎 Website | www.decodelabs.tech |
| 📍 Location | Greater Lucknow, India |

---

## 📄 License

This project is part of the DecodeLabs Industrial Training Kit, Batch 2026. All rights reserved © DecodeLabs.

---

*Built with 💻 by DecodeLabs AI Engineering Interns — Batch 2026*
