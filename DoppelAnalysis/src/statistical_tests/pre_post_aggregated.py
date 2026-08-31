#!/usr/bin/env python3
"""

Statistical analysis of pre/post survey JSONL
Run with:
    python ./src/statistical_tests/pre_post_aggregated.py
"""

from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon
from statistics import mean, stdev

# ────────────────────── CONFIG ──────────────────────
INPUT_PATH = "./data/pre_post_an_with_votes.jsonl"

SUBSCALE_QUESTIONS = {
    "CD": ["not_connect", "figure_out_who_am", "only_know_tell"],
    "DP": ["delete_when_done", "dont_keep", "fi_ask_delete"],
    "USU": ["dont_use_unless_agree", "dont_use_for_something_else", "dont_share"],
    "IA": ["companies_should_protect", "doesnt_matter_cost", "companies_do_more"],
}

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
Path("./data/results").mkdir(parents=True, exist_ok=True)


# ──────────────────── HELPERS ────────────────────────
def average_subscale(row, keys):
    try:
        pre_scores  = [row[k]["pre"] for k in keys if row[k]["pre"] is not None]
        post_scores = [row[k]["post"] for k in keys if row[k]["post"] is not None]
        if len(pre_scores) != len(keys) or len(post_scores) != len(keys):
            return None  # missing values
        return {
            "pre": round(sum(pre_scores) / len(keys), 3),
            "post": round(sum(post_scores) / len(keys), 3)
        }
    except Exception as e:
        print(f"Error in row: {e}")
        return None

def run_wilcoxon(pre_list, post_list):
    try:
        return wilcoxon(pre_list, post_list)
    except Exception as e:
        print(f"Wilcoxon error: {e}")
        return None, None

# ─────────────────────── MAIN ────────────────────────
def main():
    df = pd.read_json(INPUT_PATH, lines=True)

    for scale, questions in SUBSCALE_QUESTIONS.items():
        df[scale] = df.apply(lambda row: average_subscale(row, questions), axis=1)

        valid = df[df[scale].notnull()][scale]
        pre_values  = [x["pre"] for x in valid]
        post_values = [x["post"] for x in valid]

        stat, p = run_wilcoxon(pre_values, post_values)

        print(f"\n{scale} Subscale:")
        print(f"  N = {len(pre_values)}")
        print(f"  PRE : mean={mean(pre_values):.3f}, std={stdev(pre_values):.3f}")
        print(f"  POST: mean={mean(post_values):.3f}, std={stdev(post_values):.3f}")
        print(f"  Wilcoxon signed-rank test: p = {p:.4f}")

if __name__ == "__main__":
    main()
