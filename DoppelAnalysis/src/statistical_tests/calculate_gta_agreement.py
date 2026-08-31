
'''
How to run:
   python ./src/statistical_tests/calculate_gta_agreement.py
'''

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.file_io import load_jsonl
from utils.logging import DualWriter
Path("./data/results").mkdir(parents=True, exist_ok=True)
sys.stdout = DualWriter("./data/results/gta_agreement.txt")
from sklearn.metrics import cohen_kappa_score

def main():
    annot1 = {x["key"]: set(x["selected_ai"]) for x in \
        load_jsonl(Path("data/streamlit/annotators/guess_the_ai_annot1.jsonl"))}
    annot2 = {x["key"]: set(x["selected_ai"]) for x in \
        load_jsonl(Path("data/streamlit/annotators/guess_the_ai_annot2.jsonl"))}

    shared_keys = sorted(set(annot1) & set(annot2))

    all_bots = sorted({bot for bots in annot1.values() for bot in bots}.union(
                    {bot for bots in annot2.values() for bot in bots}))
    print(all_bots)
    labels_annot1 = []
    labels_annot2 = []
    for key in shared_keys:
        for bot in all_bots:
            labels_annot1.append(1 if bot in annot1.get(key, set()) else 0)
            labels_annot2.append(1 if bot in annot2.get(key, set()) else 0)

    score = cohen_kappa_score(labels_annot1, labels_annot2)
    print(f"\nOverall Cohen's Kappa: {score:.3f}")

if __name__ == "__main__":
    main()
