'''
python ./src/random_uniform_for_humanity.py
'''
import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import tqdm
import scipy.stats as stats

# Internal imports
from utils.file_io import load_csv, append_jsonl, load_jsonl

# -- CONFIGURATION --
DATA_PATH = "./data/final_clean_data.tsv"
MODEL_NAME = "uniform_random"
SEEDS = list(range(20))  # Averaging over 20 seeds for stability

def main():
    # --- OUTPUT PATHING ---
    out_path = f"./data/generations/{MODEL_NAME}/"
    os.makedirs(out_path, exist_ok=True)
    
    print(
        f"{'#' * 80}\n"
        f"# Humanity Labeling | Baseline: {MODEL_NAME} (Averaging over {len(SEEDS)} seeds)\n"
        f"{'#' * 80}"
    )

    # 1. LOAD DATA
    chat_data = load_csv(Path(DATA_PATH), sep="\t")
    df = pd.DataFrame(chat_data)
    
    # 2. COLLECT ALL PENDING TASKS (Static context)
    all_tasks = []
    for game_idx, game_df in df.groupby("game_idx"):
        available_rounds = sorted(game_df['round_idx'].unique())
        for i in range(len(available_rounds)):
            upto_round = available_rounds[i]
            rounds_to_include = available_rounds[:i+1]
            current_context_df = game_df[game_df['round_idx'].isin(rounds_to_include)]
            
            targets = [cn for cn in current_context_df['code_name'].unique() if cn != "GAME MASTER"]
            for target_cn in targets:
                player_rows = current_context_df[current_context_df['code_name'] == target_cn]
                player_messages = player_rows['message']

                if player_messages.empty or player_messages.str.strip().eq("").all(): continue
                if player_messages.str.contains(r"\[NON-CONSENTING\]", case=False).any(): continue

                is_actually_human = player_rows['human_bool'].iloc[0]
                all_tasks.append({
                    "game_idx": int(game_idx),
                    "upto_round": int(upto_round),
                    "target_code_name": target_cn,
                    "ground_truth_is_ai": not (str(is_actually_human).lower() == 'true')
                })

    # 3. RUN SIMULATIONS ACROSS SEEDS
    # Dictionary to store F1 lists: {round_idx: [f1_seed_0, f1_seed_1, ...]}
    round_f1_results = {0: [], 1: [], 2: []}

    for seed in tqdm(SEEDS, desc="Running Seeds"):
        random.seed(seed)
        seed_predictions = []
        
        for task in all_tasks:
            predicted_is_ai = random.choice([True, False])
            seed_predictions.append({
                **task,
                "predicted_is_ai": predicted_is_ai
            })
        
        # Calculate F1 for this specific seed
        temp_df = pd.DataFrame(seed_predictions)
        for r in [0, 1, 2]:
            round_subset = temp_df[temp_df['upto_round'] == r]
            if not round_subset.empty:
                score = f1_score(round_subset['ground_truth_is_ai'], round_subset['predicted_is_ai'])
                round_f1_results[r].append(score)

    # 4. REPORT STABILIZED RESULTS
    print(f"\n{'='*30}")
    print(f" FINAL RESULTS ({len(SEEDS)} SEEDS) ")
    print(f"{'='*30}")
    
    for r, scores in round_f1_results.items():
        if len(scores) > 1:
            n = len(scores)
            mean_f1 = np.mean(scores)
            std_err = stats.sem(scores)  # Standard Error: s / sqrt(n)
            
            # Calculate 95% CI using t-distribution
            confidence = 0.95
            h = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)
            
            print(f"Round {r} F1: {mean_f1:.3f} | 95% CI: [{mean_f1-h:.4f}, {mean_f1+h:.4f}]")

    # Optional: Save the last run for inspection
    last_run_file = Path(out_path) / "humanity_labels_last_seed.jsonl"
    for pred in seed_predictions:
        append_jsonl(str(last_run_file), pred)

    # export results as a tsv
    results_df = pd.DataFrame({
        "round": list(round_f1_results.keys()),
        "mean_f1": [np.mean(scores) for scores in round_f1_results.values()],
        "lower_ci": [np.mean(scores) - stats.sem(scores) * stats.t.ppf((1 + confidence) / 2., len(scores) - 1) for scores in round_f1_results.values()],
        "upper_ci": [np.mean(scores) + stats.sem(scores) * stats.t.ppf((1 + confidence) / 2., len(scores) - 1) for scores in round_f1_results.values()],
    })
    # round to 3 decimals
    results_df = results_df.round(3)
    results_df.to_csv(Path(out_path) / "f1_results_summary.tsv", sep="\t", index=False)
    print(f"\nResults saved to: {out_path}/f1_results_summary.tsv")
if __name__ == "__main__":
    main()