#!/usr/bin/env python3
"""
Mixed-effects analysis of pre/post Likert survey (item-level)

Run with:
    python ./src/analysis/prepost_mixed_effects.py
"""

from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

# ────────────────────── CONFIG ──────────────────────
INPUT_PATH = "./data/pre_post_an_with_votes.jsonl"
OUTPUT_PATH = "./data/results/pre_post_mixed_effects.txt"

SUBSCALE_QUESTIONS = {
    "CD": ["not_connect", "figure_out_who_am", "only_know_tell"],
    "DP": ["delete_when_done", "dont_keep", "fi_ask_delete"],
    "USU": ["dont_use_unless_agree", "dont_use_for_something_else", "dont_share"],
    "IA": ["companies_should_protect", "doesnt_matter_cost", "companies_do_more"],
}

Path("./data/results").mkdir(parents=True, exist_ok=True)

# ──────────────────── HELPERS ────────────────────────
def build_long_dataframe(df):
    """
    Convert wide JSON rows into long item-level format:
    student_id | subscale | item | time | score
    """
    rows = []

    for student_idx, row in df.iterrows():
        student_id = row.get("student_id", student_idx)

        for subscale, items in SUBSCALE_QUESTIONS.items():
            for item in items:
                cell = row.get(item)
                if not isinstance(cell, dict):
                    continue

                if cell.get("pre") is not None:
                    rows.append({
                        "student_id": student_id,
                        "subscale": subscale,
                        "item": item,
                        "time": "pre",
                        "score": cell["pre"]
                    })

                if cell.get("post") is not None:
                    rows.append({
                        "student_id": student_id,
                        "subscale": subscale,
                        "item": item,
                        "time": "post",
                        "score": cell["post"]
                    })

    return pd.DataFrame(rows)

# ─────────────────────── MAIN ────────────────────────
def main():
    df = pd.read_json(INPUT_PATH, lines=True)
    long_df = build_long_dataframe(df)

    # Encode time as numeric
    long_df["time_num"] = long_df["time"].map({"pre": 0, "post": 1})

    with open(OUTPUT_PATH, "w") as f:

        f.write("=== Mixed Effects Model: Overall Pre/Post Effect ===\n\n")

        overall_model = smf.mixedlm(
            "score ~ time_num",
            long_df,
            groups=long_df["student_id"]
        )
        overall_result = overall_model.fit()
        f.write(str(overall_result.summary()))
        f.write("\n\n")

        # ───────────── Subscale-specific models ─────────────
        for subscale in long_df["subscale"].unique():
            sub_df = long_df[long_df["subscale"] == subscale]

            f.write(f"\n=== Subscale: {subscale} ===\n\n")

            model = smf.mixedlm(
                "score ~ time_num",
                sub_df,
                groups=sub_df["student_id"]
            )
            result = model.fit()
            f.write(str(result.summary()))
            f.write("\n")

    print(f"Mixed-effects results written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()