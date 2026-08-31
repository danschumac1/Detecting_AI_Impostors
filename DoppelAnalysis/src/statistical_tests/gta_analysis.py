'''
How to run:
   python ./src/statistical_tests/gta_analysis.py
'''

from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logging import DualWriter

# --- Log to file ---
Path("./data/results").mkdir(parents=True, exist_ok=True)
sys.stdout = DualWriter("./data/results/gta_analysis.txt")


def compute_acc(gt: list, pred: list) -> float:
    gt_set = set(gt)
    pred_set = set(pred)
    return int(pred_set == gt_set)


def grade_choices(round_dict: dict) -> tuple[float, str, str]:
    ground_truth = [name for name, identity in round_dict["identities"].items() if identity != "hu"]
    prediction = round_dict["selected_ai"] + ["GAME MASTER"]
    date = round_dict["date"]
    key = round_dict["key"]
    acc = compute_acc(ground_truth, prediction)
    return acc, date, key


def evaluate_by_date(rounds: list[dict], invalid_keys: set[str]):
    scores_by_date = defaultdict(list)

    for rd in rounds:
        acc, date, key = grade_choices(rd)
        if key in invalid_keys:
            print(f"[IGNORED] Skipping invalid round {key}")
            continue
        scores_by_date[date].append(acc)

    for date in sorted(scores_by_date):
        results = scores_by_date[date]
        avg_acc = sum(results) / len(results) if results else 0.0
        print(f"{date}:")
        print(f"\tAccuracy: {avg_acc:.3f} ({len(results)} rounds)")


def main():
    adj_path = Path("./data/streamlit/annotators/guess_the_ai_adjudicated.jsonl")
    annot1_path = Path("./data/streamlit/annotators/guess_the_ai_annot1.jsonl")
    annot2_path = Path("./data/streamlit/annotators/guess_the_ai_annot2.jsonl")

    # --- List of keys to ignore ---
    invalid_keys = {
        # "301_3", "402_9", "506_3"
    }

    # --- Load Data ---
    with adj_path.open("r") as fi:
        adj = [json.loads(line) for line in fi]
    with annot1_path.open("r") as fi:
        annot1 = [json.loads(line) for line in fi]
    with annot2_path.open("r") as fi:
        annot2 = [json.loads(line) for line in fi]

    # --- Evaluate ---
    print(f"Adjudicated rounds: {len(adj)}")
    evaluate_by_date(adj, invalid_keys)

    print(f"{'*' * 20}\nannot1's rounds: {len(annot1)}")
    evaluate_by_date(annot1, invalid_keys)

    print(f"{'*' * 20}\nannot2's rounds: {len(annot2)}")
    evaluate_by_date(annot2, invalid_keys)


if __name__ == "__main__":
    main()
