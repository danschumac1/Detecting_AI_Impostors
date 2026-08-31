"""
Calculates F1 and Accuracy for humanity detection (True == AI, False == Human).
Groups metrics by round to track model performance over time.

python ./src/eval_humanity.py --model qwen --shots fs
"""
import argparse
import json
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

# Ensure this matches the file/module where we put the append_tsv function
from utils.file_io import append_tsv 

DEFAULT_OUT_PATH = "./data/results/humanity_eval_results.tsv"

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate humanity detection performance.")
    p.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH, help="Path to save the evaluation results TSV file.")
    p.add_argument("--model", type=str, required=True, choices=["gpt","llama","qwen", "uniform_random", "always_1"], help="Model name for labeling results.")
    p.add_argument("--shots", type=str, required=True, choices=["zs","fs"], help="Number of shots used in the evaluation.")
    return p.parse_args()

def main():
    args = parse_args()
    
    # Wrapped in Path() so .exists() and .name methods work correctly.
    if args.model == "uniform_random":
        input_file = Path(f"./data/generations/{args.model}/humanity_labels.jsonl")
        args.shots = "n/a"  # Uniform random doesn't use shots, so we can set this to "n/a" or similar.
    elif args.model == "always_1":
        input_file = Path(f"./data/generations/{args.model}/humanity_labels.jsonl")
        args.shots = "n/a"  # Always 1 baseline doesn't use shots, so we can set this to "n/a" or similar.
    else:     # Preserved the space after {args.shots} as requested.
        input_file = Path(f"./data/generations/{args.model}_{args.shots} /humanity_labels.jsonl")

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    data = load_jsonl(input_file)
    
    if not data:
        print(f"Error: {input_file} is empty.")
        return

    print(f"--- Humanity Detection Evaluation ({input_file.name}) ---")

    # 1. Calculate Global Metrics
    y_true_all = [item['ground_truth_is_ai'] for item in data]
    y_pred_all = [item['predicted_is_ai'] for item in data]
    
    # Initialize the dictionary with base info and global scores rounded to 3 decimals
    results_dict = {
        "model": args.model,
        "shots": args.shots,
        "acc": round(accuracy_score(y_true_all, y_pred_all), 3),
        "f1": round(f1_score(y_true_all, y_pred_all), 3)
    }

    # 2. Calculate Round-Specific Metrics (0, 1, and 2)
    for r_idx in [0, 1, 2]:
        # Filter data for the specific round
        round_data = [d for d in data if d.get('upto_round') == r_idx]
        
        if round_data:
            y_true = [item['ground_truth_is_ai'] for item in round_data]
            y_pred = [item['predicted_is_ai'] for item in round_data]
            
            results_dict[f"ut{r_idx}_acc"] = round(accuracy_score(y_true, y_pred), 3)
            results_dict[f"ut{r_idx}_f1"] = round(f1_score(y_true, y_pred), 3)
        else:
            # Handle cases where a round might be missing from the dataset
            results_dict[f"ut{r_idx}_acc"] = None
            results_dict[f"ut{r_idx}_f1"] = None

    # 3. Print out the final dictionary representation
    print("\nFinal Output Dictionary:")
    print(results_dict)

    # 4. Append to TSV
    try:
        append_tsv(args.out_path, results_dict)
        print(f"\n✅ Results successfully appended to {args.out_path}")
    except Exception as e:
        print(f"\n❌ Failed to append to TSV: {e}")

if __name__ == "__main__":
    main()