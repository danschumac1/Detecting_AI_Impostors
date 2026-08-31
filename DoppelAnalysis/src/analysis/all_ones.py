'''
python ./src/_scratch_pad/all_ones.py
'''
import sys; sys.path.append("./src")  # Ensure src/ is in the path for imports
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Internal imports (only keeping what's needed for I/O)
from utils.file_io import load_csv, append_jsonl, load_jsonl

# -- CONFIGURATION --
RESUME = True
DATA_PATH = "./data/final_clean_data.tsv"
MODEL_NAME = "always_1"

def main():
    # --- OUTPUT PATHING ---
    out_path = f"./data/generations/{MODEL_NAME}/"
    os.makedirs(out_path, exist_ok=True)
    out_file = Path(out_path) / "humanity_labels.jsonl"

    print(
        f"{'#' * 80}\n"
        f"# Humanity Labeling | Baseline: {MODEL_NAME}\n"
        f"{'#' * 80}"
    )

    # 1. LOAD DATA
    chat_data = load_csv(Path(DATA_PATH), sep="\t")
    df = pd.DataFrame(chat_data)
    
    # 2. RESUME LOGIC
    processed_keys = set()
    if RESUME and out_file.exists():
        try:
            existing = load_jsonl(out_file)
            processed_keys = {f"{i['game_idx']}_{i['upto_round']}_{i['target_code_name']}" for i in existing}
            print(f"Resuming: {len(processed_keys)} predictions already labeled.")
        except Exception as e:
            print(f"Error loading existing data: {e}. Starting fresh.")

    # 3. COLLECT ALL PENDING TASKS
    all_tasks = []
    for game_idx, game_df in tqdm(df.groupby("game_idx"), desc="Preparing Tasks"):
        available_rounds = sorted(game_df['round_idx'].unique())
        
        for i in range(len(available_rounds)):
            upto_round = available_rounds[i]
            rounds_to_include = available_rounds[:i+1]
            current_context_df = game_df[game_df['round_idx'].isin(rounds_to_include)]
            
            targets = [cn for cn in current_context_df['code_name'].unique() if cn != "GAME MASTER"]
            
            for target_cn in targets:
                res_key = f"{game_idx}_{upto_round}_{target_cn}"
                if res_key in processed_keys: 
                    continue

                player_rows = current_context_df[current_context_df['code_name'] == target_cn]
                player_messages = player_rows['message']

                if player_messages.empty or player_messages.str.strip().eq("").all(): continue
                if player_messages.str.contains(r"\[NON-CONSENTING\]", case=False).any(): continue

                is_actually_human = player_rows['human_bool'].iloc[0]
                all_tasks.append({
                    "metadata": {
                        "game_idx": int(game_idx),
                        "upto_round": int(upto_round),
                        "target_code_name": target_cn,
                        "ground_truth_is_ai": not (str(is_actually_human).lower() == 'true')
                    }
                })

    if not all_tasks:
        print("No new tasks to process. All done.")
        return

    # 4. EXECUTE BASELINE
    print(f"Processing {len(all_tasks)} evaluations using {MODEL_NAME}...")
    
    for task in tqdm(all_tasks, desc="Generating Always-1 Predictions"):
        # Explicitly assign 1 for every prediction
        predicted_is_ai = 1
        
        payload = {
            **task['metadata'],
            "predicted_is_ai": predicted_is_ai,
            "reasoning": "Always 1 baseline prediction.",
            "model": MODEL_NAME,
            "prompt_type": "none"  # Hardcoded since shots aren't applicable
        }
        append_jsonl(str(out_file), payload)

    print(f"✅ Baseline labeling complete. Results saved to {out_file}")

if __name__ == "__main__":
    main()