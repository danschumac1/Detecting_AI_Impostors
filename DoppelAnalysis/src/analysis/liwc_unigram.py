'''
How to run:
   python ./src/analysis/liwc_unigram.py
'''


import csv
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.patches as mpatches # For the color legend
import seaborn as sns

INPUT_PATH = "./data/LiwcReady.csv"

# ----------------------------
# CSV loading (no pandas)
# ----------------------------
def load_text_csv_no_pandas(
    csv_path: str,
    text_col: str = "message",
    label_col: str = "player_id",
    positive_prefix: str = "ai_",
    negative_prefix: str = "hu_",
    ignore_prefix: str = "sys_",
    drop_noconsent: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Returns:
      texts: list[str]
      y: np.ndarray int (1=AI, 0=Human)

    Rules:
      - y=1 if player_id starts with ai_
      - y=0 if player_id starts with hu_
      - ignore sys_ and any other ids
      - optionally drop rows whose message contains [NON_CONSENTING]
    """
    texts: List[str] = []
    labels: List[int] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row.")

        if text_col not in reader.fieldnames:
            raise ValueError(f"Expected text column '{text_col}' in CSV header.")
        if label_col not in reader.fieldnames:
            raise ValueError(f"Expected label column '{label_col}' in CSV header.")

        for row in reader:
            pid = (row.get(label_col) or "").strip()

            # ignore sys_ and anything not ai_/hu_
            if pid.startswith(ignore_prefix):
                continue
            if not (pid.startswith(positive_prefix) or pid.startswith(negative_prefix)):
                continue

            msg = (row.get(text_col) or "").strip()
            if drop_noconsent and "[NON_CONSENTING]" in msg:
                continue
            if msg == "":
                continue

            texts.append(msg)
            labels.append(1 if pid.startswith(positive_prefix) else 0)

    if not texts:
        raise ValueError("No AI/Human text rows found after filtering. Check prefixes/columns.")

    return texts, np.asarray(labels, dtype=np.int32)


# ----------------------------
# Top-tier ACL/EMNLP-style plot
# ----------------------------
def plot_acl_emnlp_top_terms(
    terms: List[str],
    weights: np.ndarray,
    k: int = 10,
    column: str = "two",
    xlabel: str = "Coef", # Renamed as requested
):
    # Professional research colors
    AI_COLOR = "#003f5c"   # Dark blue
    HUMAN_COLOR = "#ffa600" # Orange
    
    # Map colors to weights
    colors = [HUMAN_COLOR if w < 0 else AI_COLOR for w in weights]

    # ACL sizing
    if column == "two":
        fig_w, fig_h = 4.5, 3.45 # Swapped h/w for better vertical fit
    else:
        fig_w, fig_h = 7.0, 4.5

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
    })

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Using Seaborn for the vertical bar plot (transposed)
    sns.barplot(x=terms, y=weights, palette=colors, ax=ax, edgecolor="black", linewidth=0.5)

    # Transpose logic: Words on X, Coef on Y
    ax.set_ylabel(xlabel)
    ax.set_xlabel("Unigrams")
    plt.xticks(rotation=45, ha='right') # Rotate labels for readability

    ax.axhline(0, color="black", linewidth=0.8) # Horizontal line at zero
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.15)
    ax.set_axisbelow(True)

    sns.despine() # Clean top/right spines

    # Proper color-based legend
    human_patch = mpatches.Patch(color=HUMAN_COLOR, label='Human')
    ai_patch = mpatches.Patch(color=AI_COLOR, label='AI')
    ax.legend(handles=[human_patch, ai_patch], loc='best', frameon=False, fontsize=8)

    plt.tight_layout()
    return fig, ax

# ----------------------------
# Training: unigram counts + LR
# ----------------------------
def train_unigram_logreg_and_plot(
    csv_path: str,
    text_col: str = "message",
    label_col: str = "player_id",
    positive_prefix: str = "ai_",
    negative_prefix: str = "hu_",
    ignore_prefix: str = "sys_",
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 5000,
    k: int = 10,
    column: str = "two",
    min_df: int = 2,
    max_features: Optional[int] = 50000,
):
    """
    End-to-end:
      - load texts (no pandas), ignore sys_
      - CountVectorizer unigram counts
      - L2 LogisticRegression
      - prints eval
      - plots top-k most human (negative) + top-k most AI (positive) unigrams
    """

    texts, y = load_text_csv_no_pandas(
        csv_path=csv_path,
        text_col=text_col,
        label_col=label_col,
        positive_prefix=positive_prefix,
        negative_prefix=negative_prefix,
        ignore_prefix=ignore_prefix,
        drop_noconsent=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Simple, robust tokenization for social/chat text:
    # - keeps words & contractions (e.g., "don't")
    # - lowercases
    token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z']+\b"

    model = Pipeline([
        ("vec", CountVectorizer(
            ngram_range=(1, 1),
            lowercase=True,
            token_pattern=token_pattern,
            min_df=min_df,
            max_features=max_features,
        )),
        ("lr", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
        )),
    ])

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("Held-out performance (test split):")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

    # Extract coefficients: positive => more AI; negative => more human
    vec: CountVectorizer = model.named_steps["vec"]
    lr: LogisticRegression = model.named_steps["lr"]

    feature_names = vec.get_feature_names_out()
    coef = lr.coef_.ravel()  # shape (n_features,)

    order = np.argsort(coef)
    human_idx = order[:k]      # most negative
    ai_idx = order[-k:]        # most positive

    top_human = [(feature_names[i], float(coef[i])) for i in human_idx]
    top_ai = [(feature_names[i], float(coef[i])) for i in ai_idx]

    # Prepare plot lists in display order: human then AI
    terms = [t for t, _ in top_human] + [t for t, _ in top_ai]
    weights = np.array([w for _, w in top_human] + [w for _, w in top_ai], dtype=float)

    fig, ax = plot_acl_emnlp_top_terms(
        terms=terms,
        weights=weights,
        k=k,
        column=column,
        xlabel="Coefficient",
    )
    plt.show()

    return {
        "model": model,
        "top_human_unigrams": top_human,  # list of (term, coef) most negative
        "top_ai_unigrams": top_ai,        # list of (term, coef) most positive
        "feature_names": feature_names,
        "coefficients": coef,
        "figure": fig,
        "axis": ax,
    }




def main():
    results = train_unigram_logreg_and_plot(
        INPUT_PATH,
        text_col="message",
        k=10,
        column="two",
        min_df=2
    )
    
    # Define the filename here
    filename = "unigram_coeffs_acl.png"
    
    print("Top human:", results["top_human_unigrams"])
    print("Top AI:", results["top_ai_unigrams"])
    
    # Save using the variable
    results["figure"].savefig("unigram_coeffs_acl.pdf", bbox_inches="tight")
    results["figure"].savefig(filename, bbox_inches="tight")

    # Use the variable in the print statement
    print(f"✅ Saved png to {filename}")



if __name__ == "__main__":
    main()
