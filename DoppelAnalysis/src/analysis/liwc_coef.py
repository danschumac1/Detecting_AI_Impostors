'''
python ./src/analysis/liwc_coef.py
'''


import csv
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.patches as mpatches # For the color legend
import seaborn as sns




# ----------------------------
# CSV loading (no pandas)
# ----------------------------
def _safe_float(x: str) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_liwc_csv_no_pandas(
    csv_path: str,
    label_col: str = "player_id",
    positive_prefix: str = "ai_",
    negative_prefix: str = "hu_",
    ignore_prefix: str = "sys_",
    exclude_numeric_cols: Tuple[str, ...] = ("human_bool", "message_idx", "game_idx", "round_idx", "Segment"),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      X: (n_samples, n_features) float32
      y: (n_samples,) int (1=AI, 0=Human)
      feature_names: list of feature column names in the same order as X columns
    Notes:
      - Keeps only rows whose player_id starts with ai_ or hu_
      - Drops rows whose player_id starts with sys_
      - Uses columns that are parseable as numeric in the header, excluding exclude_numeric_cols
      - Missing/blank/non-numeric values -> 0.0
    """

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row.")

        fieldnames = list(reader.fieldnames)
        if label_col not in fieldnames:
            raise ValueError(f"Expected '{label_col}' column in CSV header.")

        # Determine candidate numeric columns from the header, excluding metadata.
        # We'll parse to floats row-by-row; any invalid value becomes 0.0.
        exclude_set = set(exclude_numeric_cols) | {label_col}
        feature_names = [c for c in fieldnames if c not in exclude_set]

        X_rows: List[List[float]] = []
        y_list: List[int] = []

        for row in reader:
            pid = (row.get(label_col) or "").strip()

            # Ignore sys_ messages (and anything else not ai_/hu_)
            if pid.startswith(ignore_prefix):
                continue
            if not (pid.startswith(positive_prefix) or pid.startswith(negative_prefix)):
                continue

            y_list.append(1 if pid.startswith(positive_prefix) else 0)

            feats = [_safe_float(row.get(c, "")) for c in feature_names]
            X_rows.append(feats)

    if not X_rows:
        raise ValueError("No AI/Human rows found after filtering. Check prefixes and input file.")

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)

    # Optional: remove any feature columns that are entirely zero (often non-numeric text columns)
    nonzero_mask = np.any(X != 0.0, axis=0)
    if not np.all(nonzero_mask):
        X = X[:, nonzero_mask]
        feature_names = [n for n, keep in zip(feature_names, nonzero_mask.tolist()) if keep]

    if X.shape[1] == 0:
        raise ValueError(
            "After filtering out empty/non-numeric columns, no usable feature columns remain. "
            "Make sure your LIWC category columns are numeric."
        )

    return X, y, feature_names



def plot_acl_emnlp_coef_bar(
    coef: np.ndarray,
    feature_names: list,
    k: int = 10,
    column: str = "two",
):
    """
    Vertical bar plot (Transposed) for LIWC categories.
    """
    # Professional color palette
    AI_COLOR = "#003f5c"   # Dark blue
    HUMAN_COLOR = "#ffa600" # Orange

    # Sort and select top k extremes
    order = np.argsort(coef)
    coef_sorted = coef[order]
    names_sorted = [feature_names[i] for i in order]

    sel_idx = list(range(0, min(k, len(coef_sorted)))) + \
              list(range(max(0, len(coef_sorted) - k), len(coef_sorted)))
    
    sel_coef = np.array([coef_sorted[i] for i in sel_idx])
    sel_names = [names_sorted[i] for i in sel_idx]
    colors = [HUMAN_COLOR if v < 0 else AI_COLOR for v in sel_coef]

    # ACL/EMNLP Sizing
    fig_width = 4.5 if column == "two" else 7.0
    fig_height = 3.5
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
    })

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Transposed Plot: x=Categories, y=Coef
    sns.barplot(x=sel_names, y=sel_coef, palette=colors, ax=ax, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Coef")
    ax.set_xlabel("LIWC Categories")
    plt.xticks(rotation=45, ha='right')

    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.1)
    sns.despine()

    # Legend using color patches
    human_patch = mpatches.Patch(color=HUMAN_COLOR, label='Human')
    ai_patch = mpatches.Patch(color=AI_COLOR, label='AI')
    ax.legend(handles=[human_patch, ai_patch], loc='best', frameon=False, fontsize=8)

    plt.tight_layout()
    return fig, ax# ----------------------------
# Training + evaluation
# ----------------------------
def train_liwc_logreg_and_plot_no_pandas(
    csv_path: str,
    label_col: str = "player_id",
    positive_prefix: str = "ai_",
    negative_prefix: str = "hu_",
    ignore_prefix: str = "sys_",
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 5000,
    k: int = 10,
) -> Dict:
    """
    End-to-end:
      - loads LIWC CSV without pandas
      - trains standardized L2 logistic regression
      - prints evaluation
      - plots k most negative and k most positive coefficients
    """
    X, y, feature_names = load_liwc_csv_no_pandas(
        csv_path=csv_path,
        label_col=label_col,
        positive_prefix=positive_prefix,
        negative_prefix=negative_prefix,
        ignore_prefix=ignore_prefix,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state
        )),
    ])

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("Held-out performance (test split):")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

    coef = model.named_steps["lr"].coef_.ravel()

    fig, ax = plot_acl_emnlp_coef_bar(
        coef=coef,
        feature_names=feature_names,
        k=k    )
    plt.show()

    # Return extremes for reporting
    order = np.argsort(coef)
    sorted_coef = coef[order]
    sorted_names = [feature_names[i] for i in order]

    top_negative = list(zip(sorted_names[:k], sorted_coef[:k]))
    top_positive = list(zip(sorted_names[-k:], sorted_coef[-k:]))

    return {
        "model": model,
        "feature_names": feature_names,
        "coefficients": coef,
        "top_negative": top_negative,  # list of (feature, coef)
        "top_positive": top_positive,  # list of (feature, coef)
        "figure": fig,
        "axis": ax,
    }



def main():
    # Use your actual data path here
    INPUT_PATH = "./data/LiwcReady.csv" 
    
    results = train_liwc_logreg_and_plot_no_pandas(
        INPUT_PATH,
        k=10
    )
    
    # Corrected Save Logic
    out_name = "liwc_coef_bar_acl_style.png"
    results["figure"].savefig("liwc_coef_bar_acl_style.pdf", bbox_inches="tight")
    results["figure"].savefig(out_name, bbox_inches="tight")
    
    print(f"✅ Saved results to {out_name}")

if __name__ == "__main__":
    main()