'''
Usage:
    python ./src/statistical_tests/calculate_acc_agreement.py
'''

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.file_io import load_jsonl
from utils.logging import DualWriter
Path("./data/results").mkdir(parents=True, exist_ok=True)
sys.stdout = DualWriter("./data/results/acc_agreement.txt")

def cohens_kappa(a, b, c, d):
    N = a + b + c + d
    po = (a + d) / N
    pe = ((a + b)*(a + c) + (c + d)*(b + d)) / (N ** 2)
    if pe == 1:  # avoid division by zero
        return 1.0
    return (po - pe) / (1 - pe)


def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

def main():
    annot1_data = load_jsonl(Path("./data/streamlit/annotators/apply_chat_codes_annot1.jsonl"))
    annot2_data = load_jsonl(Path("./data/streamlit/annotators/apply_chat_codes_annot2.jsonl"))

    annot1_dict = {line['msg_key']: set(line['labels']) for line in annot1_data}
    annot2_dict = {line['msg_key']: set(line['labels']) for line in annot2_data}

    shared_keys = sorted(set(annot1_dict.keys()) & set(annot2_dict.keys()))
    print(f"TOTAL SHARED MESSAGES: {len(shared_keys)}")

    exact_matches = 0
    jaccard_scores = []
    label_set = set()

    # Collect all unique labels
    for key in shared_keys:
        label_set |= annot1_dict[key]
        label_set |= annot2_dict[key]

    # Per-label counts
    per_label = {
        label: {
            "both_yes": 0,
            "both_no": 0,
            "annot1_yes_annot2_no": 0,
            "annot1_no_annot2_yes": 0,
        } for label in label_set
    }

    for key in shared_keys:
        annot1_labels = annot1_dict[key]
        annot2_labels = annot2_dict[key]

        if annot1_labels == annot2_labels:
            exact_matches += 1

        jaccard_scores.append(jaccard(annot1_labels, annot2_labels))

        for label in label_set:
            annot1_has = label in annot1_labels
            annot2_has = label in annot2_labels

            if annot1_has and annot2_has:
                per_label[label]["both_yes"] += 1
            elif not annot1_has and not annot2_has:
                per_label[label]["both_no"] += 1
            elif annot1_has and not annot2_has:
                per_label[label]["annot1_yes_annot2_no"] += 1
            elif not annot1_has and annot2_has:
                per_label[label]["annot1_no_annot2_yes"] += 1

    total = len(shared_keys)
    print(f"EXACT AGREEMENTS: {exact_matches} ({exact_matches/total:.2%})")
    print(f"AVERAGE JACCARD SCORE: {sum(jaccard_scores)/total:.4f}")
    print("\nPER-LABEL AGREEMENT:")

    for label, stats in sorted(per_label.items()):
        a = stats["both_yes"]
        b = stats["annot1_yes_annot2_no"]
        c = stats["annot1_no_annot2_yes"]
        d = stats["both_no"]
        total = a + b + c + d
        agree = a + d
        kappa = cohens_kappa(a, b, c, d)

        print(f"  - {label}")
        print(f"      Agreement: {agree}/{total} ({agree/total:.2%})")
        print(f"      Present in both: {a}")
        print(f"      annot1 only:        {b}")
        print(f"      annot2 only:      {c}")
        print(f"      Absent in both:  {d}")
        print(f"      Cohen's Kappa:   {kappa:.4f}")
        print()
    print(
        "Inter-rater reliability was high overall, with exact agreement on 170 out of 191 shared messages (89.01%) "
        "and an average Jaccard similarity score of 0.895.\n"
        "\n"
        "Cohen's Kappa scores indicated substantial to almost perfect agreement across most behavioral codes:\n"
        "  • 'meta': κ = 0.9175 — almost perfect agreement\n"
        "  • 'iceb': κ = 0.8790 — almost perfect agreement\n"
        "  • 'rand': κ = 0.8640 — almost perfect agreement\n"
        "  • 'crvt': κ = 0.8220 — almost perfect agreement\n"
        "  • 'folw': κ = 0.8071 — substantial agreement\n"
        "\n"
        "Agreement was weakest for 'obs' (κ = -0.0070), suggesting no reliability beyond chance. "
        "This may be due to the rarity of 'obs' labels in both annotators' data.\n"
    )


if __name__ == "__main__":
    main()
