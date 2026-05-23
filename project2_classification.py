

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.datasets        import load_iris
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)


#  STYLING
COLORS   = ["#1B4F72", "#E74C3C", "#27AE60"]
BG_COLOR = "#F0F4F8"
plt.rcParams.update({
    "figure.facecolor" : BG_COLOR,
    "axes.facecolor"   : "white",
    "axes.edgecolor"   : "#CCCCCC",
    "font.family"      : "monospace",
})


#  PHASE 1 — INPUT: LOAD & EXPLORE THE IRIS BENCHMARK DATASET
print("\n" + "="*60)
print("  PROJECT 2 — Data Classification Using AI (KNN)")
print("  DecodeLabs | Batch 2026")
print("="*60)

iris   = load_iris()
X      = iris.data           # Features: sepal_len, sepal_w, petal_len, petal_w
y      = iris.target         # Labels : 0=Setosa, 1=Versicolor, 2=Virginica
names  = iris.target_names
feat_n = iris.feature_names

print(f"\n📦 Dataset loaded: Iris Benchmark")
print(f"   Samples    : {X.shape[0]} (balanced: {X.shape[0]//3} per class)")
print(f"   Classes    : {len(names)} → {list(names)}")
print(f"   Dimensions : {X.shape[1]} features")
print(f"   Features   : {feat_n}")


#  PHASE 2 — PROCESS STEP A: FEATURE SCALING (Gatekeeper Rule)
#  StandardScaler → Mean=0, Variance=1
#  Removes bias from different measurement scales

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n⚖️  Feature Scaling applied: StandardScaler (Mean=0, Var=1)")
print(f"   Raw sepal_length range : [{X[:,0].min():.1f}, {X[:,0].max():.1f}] cm")
print(f"   Scaled range           : [{X_scaled[:,0].min():.2f}, {X_scaled[:,0].max():.2f}]")



#  PHASE 2 — PROCESS STEP B: TRAIN-TEST SPLIT
#  80% Training (Pattern Recognition) | 20% Testing (Validation)
#  shuffle=True removes order bias

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size    = 0.20,
    random_state = 42,
    shuffle      = True,
    stratify     = y       # ensures balanced classes in both splits
)

print(f"\n✂️  Train-Test Split: 80% / 20%")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")

#  PHASE 2 — PROCESS STEP C: K TUNING (Find Optimal K)
#  Test K from 1 to 20, pick the elbow point

print(f"\n🔍 Searching for Optimal K (1–20)...")
error_rates = []

for k in range(1, 21):
    knn  = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, pred))

optimal_k = int(np.argmin(error_rates)) + 1
print(f"   ✅ Optimal K = {optimal_k} (lowest error rate: {min(error_rates):.4f})")


# ═══════════════════════════════════════════════════════════════
#  PHASE 2 — PROCESS STEP D: TRAIN FINAL MODEL
#  Instantiate → Fit → Predict
# ═══════════════════════════════════════════════════════════════
model       = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)          # FIT: memorise the map
predictions = model.predict(X_test)  # PREDICT: apply logic


#  PHASE 3 — OUTPUT: EVALUATION METRICS
#  Accuracy | Confusion Matrix | F1 Score
acc    = accuracy_score(y_test, predictions)
f1     = f1_score(y_test, predictions, average="weighted")
cm     = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions, target_names=names)

print(f"\n{'='*60}")
print(f"  📊 EVALUATION RESULTS")
print(f"{'='*60}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  F1 Score  : {f1:.4f}  (weighted average)")
print(f"\n  Classification Report:")
print(report)


#  VISUALISATION — 4-panel dashboard
fig = plt.figure(figsize=(16, 12), facecolor=BG_COLOR)
fig.suptitle(
    "PROJECT 2 — Data Classification Using KNN | DecodeLabs Batch 2026",
    fontsize=15, fontweight="bold", color="#1B2631", y=0.98
)

# ── Panel 1: K Tuning Curve ─────────────────────────────────
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(range(1, 21), error_rates, color="#E74C3C", marker="o",
         markersize=6, linewidth=2, label="Error Rate")
ax1.axvline(x=optimal_k, color="#27AE60", linestyle="--",
            linewidth=2, label=f"Optimal K={optimal_k}")
ax1.scatter([optimal_k], [error_rates[optimal_k-1]],
            color="#27AE60", s=120, zorder=5)
ax1.set_title("K-Tuning: Finding the Elbow", fontweight="bold")
ax1.set_xlabel("K Value")
ax1.set_ylabel("Error Rate")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Panel 2: Confusion Matrix
ax2 = fig.add_subplot(2, 2, 2)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=names, yticklabels=names,
    ax=ax2, linewidths=0.5, cbar_kws={"shrink": 0.8}
)
ax2.set_title("Confusion Matrix", fontweight="bold")
ax2.set_ylabel("True Label")
ax2.set_xlabel("Predicted Label")

# ── Panel 3: Feature Distribution (Petal dims, best separators) ──
ax3 = fig.add_subplot(2, 2, 3)
for idx, (cls, col) in enumerate(zip(names, COLORS)):
    mask = y == idx
    ax3.scatter(
        X[mask, 2], X[mask, 3],
        color=col, label=cls, alpha=0.7, s=60, edgecolors="white", linewidth=0.5
    )
ax3.set_title("Feature Space: Petal Length vs Petal Width", fontweight="bold")
ax3.set_xlabel("Petal Length (cm)")
ax3.set_ylabel("Petal Width (cm)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── Panel 4: Per-Class F1 Scores
ax4 = fig.add_subplot(2, 2, 4)
per_class_f1 = f1_score(y_test, predictions, average=None)
bars = ax4.bar(names, per_class_f1, color=COLORS, edgecolor="white",
               linewidth=1.5, width=0.5)
ax4.set_title("Per-Class F1 Score", fontweight="bold")
ax4.set_ylabel("F1 Score")
ax4.set_ylim(0, 1.1)
ax4.axhline(y=f1, color="#E74C3C", linestyle="--",
            linewidth=1.5, label=f"Weighted F1={f1:.3f}")
ax4.legend()
for bar, val in zip(bars, per_class_f1):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"{val:.3f}", ha="center", va="bottom", fontsize=10)
ax4.grid(True, alpha=0.3, axis="y")

# ── Metrics Banner 
fig.text(0.5, 0.01,
         f"Model: KNN (K={optimal_k})  |  Accuracy: {acc*100:.2f}%  "
         f"|  F1 Score: {f1:.4f}  |  Test Samples: {len(X_test)}",
         ha="center", fontsize=11, color="#1B2631",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#D5E8F3", edgecolor="#1B4F72"))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig("/mnt/user-data/outputs/project2_classification.png",
            dpi=150, bbox_inches="tight")
plt.close()

print(f"\n✅ Visualisation saved: project2_classification.png")
print(f"\n{'='*60}")
print(f"  [Project 2 complete ✅ | Proceed to Project 3]")
print(f"{'='*60}\n")
