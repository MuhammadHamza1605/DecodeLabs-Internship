

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import math

#  THE KNOWLEDGE BASE — Job Role Catalog
#  Each role is defined by its skill "document"
#  (tags are the vocabulary of this domain)
JOB_CATALOG = {
    "Data Scientist": [
        "python", "sql", "machine learning", "statistics", "pandas",
        "numpy", "data analysis", "tensorflow", "deep learning", "visualization"
    ],
    "Machine Learning Engineer": [
        "python", "tensorflow", "pytorch", "deep learning", "machine learning",
        "mlops", "docker", "kubernetes", "api", "model deployment"
    ],
    "Data Engineer": [
        "sql", "python", "spark", "hadoop", "etl", "data pipelines",
        "aws", "azure", "kafka", "data warehousing"
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "ci/cd", "aws", "linux", "terraform",
        "ansible", "monitoring", "automation", "cloud"
    ],
    "Backend Developer": [
        "python", "java", "sql", "api", "rest", "django", "flask",
        "microservices", "docker", "databases"
    ],
    "Frontend Developer": [
        "javascript", "html", "css", "react", "typescript",
        "web design", "ui/ux", "nodejs", "vue", "angular"
    ],
    "Cloud Architect": [
        "aws", "azure", "cloud", "terraform", "kubernetes",
        "networking", "security", "microservices", "docker", "automation"
    ],
    "AI Research Scientist": [
        "python", "deep learning", "tensorflow", "pytorch", "research",
        "mathematics", "statistics", "nlp", "computer vision", "reinforcement learning"
    ],
    "Cybersecurity Engineer": [
        "linux", "networking", "security", "ethical hacking",
        "penetration testing", "firewalls", "encryption", "python", "bash", "siem"
    ],
    "Full Stack Developer": [
        "javascript", "python", "react", "nodejs", "sql", "html",
        "css", "api", "docker", "databases"
    ],
    "NLP Engineer": [
        "python", "nlp", "tensorflow", "pytorch", "transformers",
        "bert", "text processing", "machine learning", "deep learning", "linguistics"
    ],
    "Database Administrator": [
        "sql", "mysql", "postgresql", "oracle", "databases",
        "data warehousing", "backup", "performance tuning", "replication", "indexing"
    ],
}

#  TF-IDF ENGINE  (from scratch — no sklearn)
#  Term Frequency × Inverse Document Frequency
#  Rewards specific terms, penalizes generic ones

def build_vocabulary(catalog: dict) -> list:
    """Collect every unique skill across all job roles."""
    vocab = set()
    for skills in catalog.values():
        vocab.update(skills)
    return sorted(list(vocab))


def compute_tf(document: list) -> dict:
    """Term Frequency: count(term) / total_terms in document."""
    tf = defaultdict(float)
    total = len(document)
    for term in document:
        tf[term] += 1.0 / total
    return tf


def compute_idf(catalog: dict, vocabulary: list) -> dict:
    """
    Inverse Document Frequency:
    log(total_docs / docs_containing_term)
    Penalises terms that appear in too many roles (generic).
    """
    N   = len(catalog)
    idf = {}
    for term in vocabulary:
        docs_with_term = sum(1 for skills in catalog.values() if term in skills)
        idf[term] = math.log(N / (docs_with_term + 1e-9))  # smoothing
    return idf


def compute_tfidf_vector(document: list, idf: dict, vocabulary: list) -> np.ndarray:
    """Build a TF-IDF weighted vector for a single document."""
    tf  = compute_tf(document)
    vec = np.array([tf.get(term, 0.0) * idf.get(term, 0.0) for term in vocabulary])
    return vec


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    cos(θ) = (A · B) / (||A|| × ||B||)
    Range: 0 (orthogonal) → 1 (perfectly aligned)
    Invariant to magnitude — focuses on orientation.
    """
    dot     = np.dot(vec_a, vec_b)
    norm_a  = np.linalg.norm(vec_a)
    norm_b  = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Cold Start protection
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────
#  PRE-COMPUTE: Build TF-IDF vectors for all roles
# ─────────────────────────────────────────────
VOCABULARY      = build_vocabulary(JOB_CATALOG)
IDF             = compute_idf(JOB_CATALOG, VOCABULARY)
ROLE_VECTORS    = {
    role: compute_tfidf_vector(skills, IDF, VOCABULARY)
    for role, skills in JOB_CATALOG.items()
}


#  THE 4-STEP RANKING PIPELINE
#  Step 1: Ingest   — capture user profile
#  Step 2: Score    — cosine similarity vs all roles
#  Step 3: Sort     — descending by score
#  Step 4: Filter   — Top-N to prevent choice overload

def recommend(user_skills: list, top_n: int = 3) -> list:
    """
    Core recommendation engine.
    Returns Top-N (role, score) tuples sorted by cosine similarity.
    """
    # ── Step 1: INGEST
    # Normalize user input to match our vocabulary
    user_skills_clean = [s.lower().strip() for s in user_skills]

    # Cold Start guard: if no matching skills, fallback
    known = [s for s in user_skills_clean if s in VOCABULARY]
    if not known:
        return [("⚠️ Cold Start", 0.0,
                 "No matching skills found. Try: python, sql, docker, aws, etc.")]

    # Build user profile vector (treated as a 1-doc TF-IDF)
    user_vector = compute_tfidf_vector(user_skills_clean, IDF, VOCABULARY)

    # ── Step 2: SCORE
    scored = [
        (role, cosine_similarity(user_vector, role_vec))
        for role, role_vec in ROLE_VECTORS.items()
    ]

    # ── Step 3: SORT
    scored.sort(key=lambda x: x[1], reverse=True)

    # ── Step 4: FILTER
    return [(role, round(score, 4)) for role, score in scored[:top_n]]


#  VISUALISATION — Recommendation Dashboard

def plot_results(user_skills: list, results: list, output_path: str):
    """Generate a professional 3-panel recommendation dashboard."""
    BG   = "#F7F9FC"
    DARK = "#1B2631"

    fig  = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle(
        "PROJECT 3 — AI Recommendation Logic (Tech Stack Recommender)\n"
        "DecodeLabs | Batch 2026 | Algorithm: TF-IDF + Cosine Similarity",
        fontsize=13, fontweight="bold", color=DARK, y=0.99
    )

    # ── Panel A: All roles ranked (full bar chart) 
    ax1 = fig.add_subplot(1, 3, (1, 2))

    # Score all roles for the full ranking view
    user_vector = compute_tfidf_vector(
        [s.lower().strip() for s in user_skills], IDF, VOCABULARY
    )
    all_scores = sorted(
        [(role, cosine_similarity(user_vector, rv)) for role, rv in ROLE_VECTORS.items()],
        key=lambda x: x[1]
    )
    roles_sorted  = [r for r, _ in all_scores]
    scores_sorted = [s for _, s in all_scores]

    cmap     = plt.colormaps.get_cmap("RdYlGn")
    max_sc   = max(scores_sorted) if max(scores_sorted) > 0 else 1
    bar_cols = [cmap(s / max_sc) for s in scores_sorted]

    bars = ax1.barh(roles_sorted, scores_sorted, color=bar_cols,
                    edgecolor="white", linewidth=0.8)

    # Highlight Top-N
    top_roles = {r for r, _ in results}
    for bar, role in zip(bars, roles_sorted):
        if role in top_roles:
            bar.set_edgecolor("#E74C3C")
            bar.set_linewidth(2.5)

    ax1.set_title(f"All Roles Ranked by Cosine Similarity\n"
                  f"User Skills: {', '.join(user_skills)}",
                  fontweight="bold", fontsize=11, color=DARK)
    ax1.set_xlabel("Cosine Similarity Score (0=No Match, 1=Perfect)")
    ax1.set_xlim(0, max_sc * 1.25)
    ax1.axvline(x=max_sc * 0.5, color="grey", linestyle="--", alpha=0.5, linewidth=1)

    for bar, score in zip(bars, scores_sorted):
        ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{score:.3f}", va="center", fontsize=9, color=DARK)
    ax1.grid(True, alpha=0.2, axis="x")

    # ── Panel B: Top-N Donut Chart ────────────────────────────
    ax2 = fig.add_subplot(1, 3, 3)
    top_labels = [r for r, _ in results]
    top_scores = [s for _, s in results]

    if sum(top_scores) > 0:
        pie_colors = ["#1B4F72", "#2E86C1", "#85C1E9"][:len(results)]
        wedges, texts, autotexts = ax2.pie(
            top_scores,
            labels    = None,
            autopct   = "%1.1f%%",
            colors    = pie_colors,
            startangle= 90,
            wedgeprops= dict(width=0.55, edgecolor="white", linewidth=2),
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_color("white")
            at.set_fontweight("bold")

        legend_labels = [f"{r}\n({s:.3f})" for r, s in results]
        ax2.legend(wedges, legend_labels,
                   loc="lower center", bbox_to_anchor=(0.5, -0.25),
                   fontsize=9, frameon=False)
    else:
        ax2.text(0.5, 0.5, "No Match\n(Cold Start)",
                 ha="center", va="center", fontsize=14, color="#E74C3C")

    ax2.set_title(f"Top {len(results)} Recommendations",
                  fontweight="bold", fontsize=11, color=DARK)

    # ── Footer ────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "Pipeline: Ingest → TF-IDF Vectorize → Cosine Score → Sort → Filter  |  "
        "Red borders = Top recommendations",
        ha="center", fontsize=9, color=DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#D5E8F3", edgecolor="#1B4F72")
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


#  INTERACTIVE SESSION

def run_recommender():
    print("\n" + "="*62)
    print("  PROJECT 3 — AI Recommendation Logic")
    print("  System: Tech Stack Recommender | DecodeLabs Batch 2026")
    print("  Algorithm: TF-IDF + Cosine Similarity")
    print("="*62)
    print("\n  Available skill keywords (sample):")
    sample = ["python", "sql", "docker", "aws", "machine learning",
              "javascript", "deep learning", "react", "kubernetes", "nlp"]
    print(f"  {', '.join(sample)} ... and more\n")

    # ── INGESTION STEP: minimum 3 inputs
    user_skills = []
    print("  Enter at least 3 skills (one per line). Press Enter twice when done.")
    print("─"*62)
    count = 0
    while True:
        skill = input(f"  Skill {count+1}: ").strip().lower()
        if skill == "" and count >= 3:
            break
        elif skill == "" and count < 3:
            print(f"  ⚠️  Please enter at least 3 skills ({count} so far).")
            continue
        user_skills.append(skill)
        count += 1

    top_n = 3
    try:
        n_input = input(f"\n  How many recommendations? [default=3]: ").strip()
        if n_input:
            top_n = max(1, min(int(n_input), len(JOB_CATALOG)))
    except ValueError:
        pass

    print("\n" + "─"*62)
    print(f"  🔍 Running 4-Step Pipeline for: {user_skills}")
    print("─"*62)

    results = recommend(user_skills, top_n=top_n)

    print(f"\n  🎯 TOP {top_n} RECOMMENDED ROLES:")
    print("─"*62)
    for rank, (role, score) in enumerate(results, 1):
        match_pct = score * 100
        bar_len   = int(match_pct * 0.4)
        bar       = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  #{rank} {role}")
        print(f"      [{bar}] {match_pct:.1f}% match")
        print()

    # Overlapping skills breakdown
    print("  📋 Skill Overlap Analysis:")
    for role, score in results:
        catalog_skills = set(JOB_CATALOG.get(role, []))
        user_set       = set(user_skills)
        overlap        = catalog_skills & user_set
        missing        = catalog_skills - user_set
        print(f"\n  {role}:")
        print(f"    ✅ Matched : {', '.join(overlap) if overlap else 'None'}")
        print(f"    📌 To Learn: {', '.join(list(missing)[:4]) if missing else 'You cover all!'}")

    # Plot
    plot_results(
        user_skills, results,
        "/mnt/user-data/outputs/project3_recommendation.png"
    )
    print(f"\n  ✅ Visualisation saved: project3_recommendation.png")
    print("\n" + "="*62)
    print("  [Project 3 complete ✅ | Proceed to Project 4]")
    print("="*62 + "\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_recommender()
