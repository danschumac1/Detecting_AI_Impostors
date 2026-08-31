'''
How to run:
   python ./src/statistical_tests/second_chat_codes_analysis.py
'''

import json
from pathlib import Path
import sys
from typing import List
from collections import Counter
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency, fisher_exact


Path("./data/results").mkdir(parents=True, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fi:
        return [json.loads(line) for line in fi]

def count_labels(data: List[dict]) -> Counter:
    counter = Counter()
    for line in data:
        labels = line.get("labels", [])
        for label in labels:
            counter[label.strip().lower()] += 1
    return counter

def run_code_tests(counts1: Counter, counts2: Counter, label="Label"):
    all_keys = sorted(set(counts1.keys()) | set(counts2.keys()))
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())

    print(f"\n🧪 Statistical tests for {label}s")
    print(f"{'Code':>10} | {'Test':>6} | {'p-value':>8} | {'D1 %':>6} | {'D2 %':>6} | {'Δ':>6}")
    print("-" * 60)

    for key in all_keys:
        a = counts1.get(key, 0)
        c = counts2.get(key, 0)
        b = total1 - a
        d = total2 - c
        table = [[a, b], [c, d]]

        test = fisher_exact if any(x < 5 for row in table for x in row) else chi2_contingency
        test_name = "fisher" if test is fisher_exact else "chi2"

        p = test(table)[1]
        prop1 = a / total1
        prop2 = c / total2
        delta = prop2 - prop1

        print(f"{key:>10} | {test_name:>6} | {p:>8.4f} | {prop1:>6.1%} | {prop2:>6.1%} | {delta:+.3f}")

# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    data = load_jsonl(Path("./data/streamlit/annotators/acc_adj_cleaned.jsonl"))
    day1 = [line for line in data if line["date"] in ["6-09", "06-09"]]
    day2 = [line for line in data if line["date"] in ["6-10", "06-10"]]

    label_counts_d1 = count_labels(day1)
    label_counts_d2 = count_labels(day2)

    print("="*60)
    print("TEST: Comparing the distribution of label codes across Day 1 and Day 2 using chi-square or Fisher’s exact test.")
    print("WHY: Label codes represent categorical behaviors. We want to know if the proportion of each code differs significantly across days.")
    print("ASSUMPTIONS: Messages are independently labeled. Chi-square is used for common labels; Fisher’s exact is used for rare labels.")
    
    run_code_tests(label_counts_d1, label_counts_d2, label="Label")

    print("\nTAKE AWAY:")
    print("Out of the six behavioral codes analyzed, three showed statistically significant "
      "changes across days. The proportion of 'rand' (random or off-topic messages) "
      "decreased by 28.8 percentage points from Day 1 to Day 2 (p < 0.0001), suggesting "
      "that students produced fewer chaotic or irrelevant responses as they became more "
      "familiar with the game. The use of 'meta' strategies—messages that referenced "
      "human-only knowledge or exploited AI limitations—increased by 12.4 percentage points "
      "(p = 0.0001), indicating a growing awareness of the game’s mechanics. Finally, 'crvt' "
      "(creative group coordination) increased by 11.8 percentage points (p = 0.0156), reflecting "
      "a rise in messages that aimed to organize or influence voting. The remaining codes "
      "('folw', 'obs', 'iceb') did not show statistically significant differences. Overall, the "
      "results suggest a behavioral shift toward more deliberate and strategic communication on "
      "Day 2.")



if __name__ == "__main__":
    main()
