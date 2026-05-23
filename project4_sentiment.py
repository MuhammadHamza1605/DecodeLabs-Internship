"""
============================================================
  DecodeLabs | AI Industrial Training Kit | Batch 2026
  PROJECT 4: AI-Powered Sentiment Analysis
  Goal    : Classify text as Positive / Negative / Neutral
  Method  : NLP pipeline — TF-IDF + Logistic Regression
  Skills  : NLP, text preprocessing, multi-class
            classification, real-world AI deployment pattern
============================================================
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.linear_model        import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection     import train_test_split
from sklearn.metrics             import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ═══════════════════════════════════════════════════════════════
#  TRAINING CORPUS  — 90 balanced samples (30 per class)
# ═══════════════════════════════════════════════════════════════
DATASET = [
    # ── POSITIVE (30) ─────────────────────────────────────────
    ("This product is amazing best purchase I ever made", "positive"),
    ("Fantastic quality and super fast delivery highly recommended", "positive"),
    ("I love this it works perfectly exceeded all my expectations", "positive"),
    ("Outstanding performance the team did an incredible job", "positive"),
    ("Very happy with my experience will definitely buy again", "positive"),
    ("The AI course at DecodeLabs is brilliant learned so much", "positive"),
    ("Great value for money works exactly as described", "positive"),
    ("Excellent customer support and a wonderful product", "positive"),
    ("This made my life so much easier absolutely brilliant", "positive"),
    ("Top notch quality fast shipping and beautiful packaging", "positive"),
    ("The project was completed flawlessly superb engineering", "positive"),
    ("I am thrilled with the results everything is perfect", "positive"),
    ("Wonderful experience from start to finish ten out of ten", "positive"),
    ("The best software I have used in years highly efficient", "positive"),
    ("Exceeded expectations in every way very impressed", "positive"),
    ("Absolutely love this product recommend to everyone", "positive"),
    ("Superb quality and amazing price worth every penny", "positive"),
    ("Delightful experience will come back again and again", "positive"),
    ("The results are phenomenal beyond my wildest expectations", "positive"),
    ("Brilliant work by the team everything is perfect", "positive"),
    ("So happy with this purchase exceeded all my hopes", "positive"),
    ("Incredible value fast delivery and stunning quality", "positive"),
    ("Highly recommend this to anyone looking for quality", "positive"),
    ("Best decision I made buying this product absolutely stellar", "positive"),
    ("The team delivered outstanding results I am very impressed", "positive"),
    ("Wonderful quality and excellent service five stars", "positive"),
    ("Perfect product perfect price perfect overall experience", "positive"),
    ("This is exactly what I needed amazing job everyone", "positive"),
    ("Very satisfied with the purchase everything works great", "positive"),
    ("Top quality product fantastic customer service loved it", "positive"),
    # ── NEGATIVE (30) ─────────────────────────────────────────
    ("This is the worst product I have ever bought complete waste", "negative"),
    ("Terrible quality broke after one day very disappointed", "negative"),
    ("Horrible experience customer service was completely useless", "negative"),
    ("Do not buy this it is a total scam absolute garbage", "negative"),
    ("Extremely poor quality nothing like the description at all", "negative"),
    ("Awful the item arrived damaged and support never responded", "negative"),
    ("I regret buying this total waste of money and time", "negative"),
    ("Pathetic performance the AI model keeps giving wrong answers", "negative"),
    ("Disgusting service I waited three weeks for nothing", "negative"),
    ("This product is defective and dangerous avoid at all costs", "negative"),
    ("Worst customer experience ever no refund and no response", "negative"),
    ("Very frustrating the software crashes every few minutes", "negative"),
    ("Overpriced and underperforming not worth a single penny", "negative"),
    ("Failed completely I am furious about this terrible purchase", "negative"),
    ("Shockingly bad everything about this product is wrong", "negative"),
    ("Complete disaster broken on arrival and terrible support", "negative"),
    ("Do not waste your money on this awful useless product", "negative"),
    ("Ridiculous quality my expectations were completely wrong", "negative"),
    ("I hate this product it ruined my entire day", "negative"),
    ("Unacceptable performance nothing as advertised stay away", "negative"),
    ("The worst experience of my life totally useless", "negative"),
    ("Defective garbage poor materials terrible packaging", "negative"),
    ("Absolute nightmare to use broken and incredibly frustrating", "negative"),
    ("Money down the drain useless junk avoid completely", "negative"),
    ("Disgusted by the quality will never buy from them again", "negative"),
    ("Horrible product and terrible company stay far away", "negative"),
    ("Utter disappointment worst purchase I have ever made", "negative"),
    ("Nothing works correctly total waste of money avoid this", "negative"),
    ("Furious about this terrible experience will never return", "negative"),
    ("Broken defective and the company refuses to help awful", "negative"),
    # ── NEUTRAL (30) ──────────────────────────────────────────
    ("The package arrived on time nothing special about it", "neutral"),
    ("It is an average product does the job nothing more", "neutral"),
    ("The item is okay not great not terrible just mediocre", "neutral"),
    ("I received the product it looks fine for now", "neutral"),
    ("The course covers the basics it is what it is", "neutral"),
    ("The delivery took seven days the product is functional", "neutral"),
    ("It is a standard model with typical average features", "neutral"),
    ("The project was completed on schedule nothing notable", "neutral"),
    ("The AI output was generated successfully results are pending", "neutral"),
    ("The software installed correctly performance is average", "neutral"),
    ("The report has been submitted and is awaiting review", "neutral"),
    ("The device powers on battery life seems normal", "neutral"),
    ("The training data was loaded processing has begun now", "neutral"),
    ("The system is operational no errors detected today", "neutral"),
    ("The update was applied no significant changes observed", "neutral"),
    ("Received the item as described nothing unexpected at all", "neutral"),
    ("Works as expected standard quality nothing impressive", "neutral"),
    ("The product functions correctly no issues detected so far", "neutral"),
    ("Average performance meets the minimum requirements only", "neutral"),
    ("Standard delivery timeline product is operational and fine", "neutral"),
    ("The results are within expected range nothing unusual", "neutral"),
    ("Basic functionality is present nothing outstanding here", "neutral"),
    ("The item arrived in acceptable condition as described", "neutral"),
    ("Meets the requirements without any issues or highlights", "neutral"),
    ("The process completed without errors no special notes", "neutral"),
    ("Standard product with normal average performance metrics", "neutral"),
    ("Order confirmed and delivered on the expected date", "neutral"),
    ("The tool works for the intended purpose nothing more", "neutral"),
    ("Status is normal operations are running as expected", "neutral"),
    ("The model returned a result processing was completed", "neutral"),
]

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
LABEL_MAP    = {"positive": 2, "neutral": 1, "negative": 0}
LABEL_NAMES  = ["negative", "neutral", "positive"]
LABEL_COLORS = {"positive": "#27AE60", "neutral": "#F39C12", "negative": "#E74C3C"}
LABEL_EMOJIS = {"positive": "😊", "neutral": "😐", "negative": "😞"}

POS_LEXICON = {"amazing","fantastic","excellent","brilliant","love","great",
               "wonderful","outstanding","superb","perfect","thrilled","happy",
               "best","recommended","impressive","stellar","phenomenal"}
NEG_LEXICON = {"terrible","worst","horrible","awful","disgusting","useless",
               "garbage","defective","pathetic","frustrating","scam","broken",
               "waste","furious","regret","disaster","nightmare","hatred"}


# ═══════════════════════════════════════════════════════════════
#  NLP PREPROCESSING
# ═══════════════════════════════════════════════════════════════
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|\S+@\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lexicon_signal(text: str) -> str:
    words    = set(text.lower().split())
    pos_hits = len(words & POS_LEXICON)
    neg_hits = len(words & NEG_LEXICON)
    if pos_hits > neg_hits:   return "positive"
    elif neg_hits > pos_hits: return "negative"
    return "neutral"


# ═══════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════
def train_model():
    print("\n" + "="*62)
    print("  PROJECT 4 — AI Sentiment Analysis")
    print("  DecodeLabs | Batch 2026 | NLP Pipeline")
    print("="*62)

    texts  = [preprocess(t) for t, _ in DATASET]
    labels = [LABEL_MAP[l] for _, l in DATASET]

    print(f"\n📦 Corpus: {len(texts)} samples")
    dist = Counter(l for _, l in DATASET)
    for sent in ["positive", "neutral", "negative"]:
        print(f"   {LABEL_EMOJIS[sent]} {sent.capitalize():10s}: {dist[sent]} samples")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range  = (1, 2),
        max_features = 1000,
        sublinear_tf = True,
        min_df       = 1,
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    print(f"\n🔤 TF-IDF Vectorizer:")
    print(f"   Vocabulary : {len(vectorizer.vocabulary_)} terms")
    print(f"   N-grams    : (1,2) — unigrams + bigrams")
    print(f"   Matrix     : {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n✂️  Split 80/20 → Train:{X_train.shape[0]} | Test:{X_test.shape[0]}")

    # Logistic Regression
    clf = LogisticRegression(
        solver='lbfgs', max_iter=500, C=2.0, random_state=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted")
    cm  = confusion_matrix(y_test, preds)
    rep = classification_report(y_test, preds, target_names=LABEL_NAMES)

    print(f"\n{'='*62}")
    print(f"  📊 EVALUATION RESULTS")
    print(f"{'='*62}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  F1 Score  : {f1:.4f} (weighted)")
    print(f"\n  Classification Report:")
    print(rep)

    return clf, vectorizer, acc, f1, cm, X_test, y_test, preds


# ═══════════════════════════════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════════════════════════════
def predict_sentiment(text: str, clf, vectorizer) -> dict:
    cleaned  = preprocess(text)
    vec      = vectorizer.transform([cleaned])
    pred_idx = clf.predict(vec)[0]
    proba    = clf.predict_proba(vec)[0]
    label    = LABEL_NAMES[pred_idx]
    return {
        "label"      : label,
        "emoji"      : LABEL_EMOJIS[label],
        "confidence" : float(proba[pred_idx]),
        "all_scores" : {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)},
        "lexicon"    : lexicon_signal(text),
    }


# ═══════════════════════════════════════════════════════════════
#  VISUALISATION DASHBOARD
# ═══════════════════════════════════════════════════════════════
def plot_dashboard(acc, f1, cm, y_test, preds, sample_results, output_path):
    BG   = "#F7F9FC"
    DARK = "#1B2631"

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        "PROJECT 4 — AI Sentiment Analysis | DecodeLabs Batch 2026\n"
        "Algorithm: TF-IDF Bigrams + Logistic Regression",
        fontsize=13, fontweight="bold", color=DARK, y=0.99
    )

    # Panel 1: Confusion Matrix
    ax1 = fig.add_subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                ax=ax1, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax1.set_title("Confusion Matrix", fontweight="bold")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    # Panel 2: Actual vs Predicted
    ax2 = fig.add_subplot(2, 3, 2)
    true_c = Counter(y_test)
    pred_c = Counter(preds)
    x = np.arange(len(LABEL_NAMES))
    w = 0.35
    ax2.bar(x-w/2, [true_c.get(i,0) for i in range(3)], width=w,
            label="Actual",    color=["#E74C3C","#F39C12","#27AE60"], alpha=0.85)
    ax2.bar(x+w/2, [pred_c.get(i,0) for i in range(3)], width=w,
            label="Predicted", color=["#C0392B","#D68910","#1E8449"], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABEL_NAMES)
    ax2.set_title("Actual vs Predicted", fontweight="bold")
    ax2.set_ylabel("Count")
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Per-class F1
    ax3 = fig.add_subplot(2, 3, 3)
    pcf = f1_score(y_test, preds, average=None)
    cols = [LABEL_COLORS[n] for n in LABEL_NAMES]
    bars = ax3.bar(LABEL_NAMES, pcf, color=cols, edgecolor="white", width=0.5)
    ax3.set_title("Per-Class F1 Score", fontweight="bold")
    ax3.set_ylabel("F1 Score"); ax3.set_ylim(0, 1.15)
    ax3.axhline(y=f1, linestyle="--", color=DARK, linewidth=1.5,
                label=f"Weighted F1={f1:.3f}")
    ax3.legend(fontsize=9)
    for bar, val in zip(bars, pcf):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f"{val:.3f}", ha="center", fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4 (bottom): Confidence bars for sample predictions
    ax4 = fig.add_subplot(2, 1, 2)
    if sample_results:
        n = len(sample_results)
        y_pos = np.arange(n)
        bw    = 0.25
        for j, (sent, col) in enumerate(zip(LABEL_NAMES, cols)):
            vals = [r["all_scores"].get(sent, 0) for r in sample_results]
            ax4.barh(y_pos + j*bw, vals, height=bw,
                     color=col, label=sent.capitalize(), alpha=0.85)
        ax4.set_yticks(y_pos + bw)
        ytl = [
            f"#{i+1} {r['emoji']} {r['label'].upper()} ({r['confidence']*100:.0f}%)"
            for i, r in enumerate(sample_results)
        ]
        ax4.set_yticklabels(ytl, fontsize=9)
        ax4.set_xlabel("Probability Score")
        ax4.set_title("Sample Predictions — Confidence per Class", fontweight="bold")
        ax4.legend(loc="lower right")
        ax4.axvline(x=0.5, color="grey", linestyle="--", alpha=0.4)
        ax4.grid(True, alpha=0.2, axis="x")

    fig.text(0.5, 0.01,
             f"Model: Logistic Regression | TF-IDF Bigrams  |  "
             f"Accuracy: {acc*100:.2f}%  |  F1: {f1:.4f}  |  "
             f"Corpus: 90 samples (30 per class)",
             ha="center", fontsize=10, color=DARK,
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#D5E8F3", edgecolor="#1B4F72"))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  INTERACTIVE ANALYSER
# ═══════════════════════════════════════════════════════════════
DEMO = [
    "This AI course is absolutely fantastic I learned so much!",
    "The product broke after two days complete waste of money.",
    "The package has been delivered it is what I ordered.",
    "I am incredibly frustrated nothing works as advertised.",
    "Decent performance overall nothing extraordinary.",
]

def run_analyser(clf, vectorizer):
    print("\n" + "─"*62)
    print("  🧠 LIVE SENTIMENT ANALYSER")
    print("  Commands: 'demo' → run 5 demo sentences | 'exit' → quit")
    print("─"*62 + "\n")

    sample_results = []
    while True:
        user_input = input("  Enter text: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "demo":
            print("\n  ── 5 Demo Sentences ──")
            for sent in DEMO:
                result = predict_sentiment(sent, clf, vectorizer)
                sample_results.append(result)
                bar = "█" * int(result["confidence"]*20)
                print(f"\n  📝 {sent[:55]}...")
                print(f"  {result['emoji']} {result['label'].upper():8s} | "
                      f"[{bar:<20}] {result['confidence']*100:.1f}%")
                print(f"  Pos={result['all_scores']['positive']:.3f} | "
                      f"Neu={result['all_scores']['neutral']:.3f} | "
                      f"Neg={result['all_scores']['negative']:.3f} | "
                      f"Lexicon={result['lexicon'].upper()}")
            continue

        result = predict_sentiment(user_input, clf, vectorizer)
        sample_results.append(result)
        bar = "█" * int(result["confidence"]*20)
        print(f"\n  {'─'*50}")
        print(f"  {result['emoji']} Sentiment : {result['label'].upper()}")
        print(f"  Confidence : [{bar:<20}] {result['confidence']*100:.1f}%")
        print(f"  Scores     : Pos={result['all_scores']['positive']:.3f} | "
              f"Neu={result['all_scores']['neutral']:.3f} | "
              f"Neg={result['all_scores']['negative']:.3f}")
        print(f"  Lexicon    : {result['lexicon'].upper()}")
        print(f"  {'─'*50}\n")

    return sample_results


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    clf, vectorizer, acc, f1, cm, X_test, y_test, preds = train_model()
    sample_results = run_analyser(clf, vectorizer)
    print("\n  📊 Generating dashboard...")
    plot_dashboard(acc, f1, cm, y_test, preds,
                   sample_results[:5] if sample_results else [],
                   "/mnt/user-data/outputs/project4_sentiment.png")
    print("  ✅ Saved: project4_sentiment.png")
    print("\n" + "="*62)
    print("  [Project 4 complete ✅ | All 4 Projects Done! 🎉]")
    print("="*62 + "\n")
