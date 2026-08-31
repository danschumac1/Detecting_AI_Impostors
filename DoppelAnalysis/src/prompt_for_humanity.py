
'''
see ./bin/generation/prompt_for_humanity.sh 
'''

import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, Field

# Internal imports
from utils.prompters import OpenAIPrompter, QwenPrompter, LlamaPrompter, Prompter
from utils.prompt_objects import FullPrompt, PromptTurn
from utils.file_io import load_csv, append_jsonl, load_jsonl

# -- CONFIGURATION --
RESUME = True
DATA_PATH = "./data/final_clean_data.tsv"

GAME_RULES = (
    "You are one of six players in a social deduction game taking place in a chatroom. "
    "Of the six players, three are humans (including you), and three are AI doppelgängers. "
    "The AI players are designed to mimic real humans by copying names, interests, and writing styles. "
    "The humans win if they vote out all AI players. The AIs win if they outnumber the humans."
)

class HumanityOutput(BaseModel):
    is_AI: bool = Field(description="True if the entity is an AI, False if it is a human.")
    reasoning: str = Field(description="Brief reasoning for the classification.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Humanity prediction with specified model.")
    parser.add_argument("--model_type", type=str, choices=["gpt", "qwen", "llama"], required=True)
    parser.add_argument("--model_id", type=str, default=None, help="Specific model ID to use (overrides defaults).")
    parser.add_argument("--shots", type=str, choices=["zs", "fs"], required=True, help="Zero-shot (zs) or Few-shot (fs)")
    parser.add_argument("--batch_size", type=int, default=15, help="Batch size for local or max_workers for API")
    parser.add_argument("--debug", type=int, default=0, choices=[0,1], help="Enable verbose debug prints")
    return parser.parse_args()

def get_prompter(model_type: str, model_id: str, debug: bool):
    if model_type == "gpt":
        model = model_id or "gpt-4o-mini"
        print(f"Initializing OpenAI Prompter with {model}...")
        return OpenAIPrompter(model_id=model), model
    elif model_type == "qwen":
        model = model_id or "Qwen/Qwen3.5-27B"
        print(f"Initializing Qwen Prompter (Local) with {model}...")
        return QwenPrompter(model_id=model, debug=bool(debug)), model
    elif model_type == "llama":
        model = model_id or "meta-llama/Llama-3.1-8B-Instruct"
        print(f"Initializing Llama Prompter (Local) with {model}...")
        return LlamaPrompter(model_id=model, debug=bool(debug)), model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    args = parse_args()

    # --- OUTPUT PATHING ---
    out_path = f"./data/generations/{args.model_type}_{args.shots}/"
    os.makedirs(out_path, exist_ok=True)
    out_file = Path(out_path) / f"humanity_labels.jsonl"

    print(
        f"{'#' * 80}\n"
        f"# Humanity Labeling | {args.model_type} | {args.shots} | Batch: {args.batch_size}\n"
        f"{'#' * 80}"
    )

    # --- PROMPTER SETUP ---
    prompter, active_model_id = get_prompter(args.model_type, args.model_id, args.debug)

    system_msg = (
        "You will be given a transcript of a social deduction game and a specific code name. "
        "Determine if the player behind that code name is an AI or a Human. "
        f"The players see the following game rules: ```{GAME_RULES}```\n\n"
        "Note: Messages replaced with [NON-CONSENTING] are for context only; do not label them."
    )    

    # Setup Few-Shot or Zero-Shot Examples
    examples = []
    if args.shots == "fs":
        examples = [
            PromptTurn(
                user_input={
                    "transcript": "GAME MASTER: Welcome to the game! Everyone please introduce yourself using your first name and last initial.\nLION: yo im Alex M\nZEBRA: [NON-CONSENTING]\nMONKEY: Hey everyone, Jessica R here.\nSNAKE: Greetings, I am Alex M as well.\nLION: wait snake why did u copy my name\nHORSE: [NON-CONSENTING]\nTIGER: im Sam K\nSNAKE: I did not copy you. I am the real Alex M. I enjoy hiking and video games.\nLION: lmao okay bot.\nZEBRA: [NON-CONSENTING]\nMONKEY: snake sounds suspiciously formal\nSNAKE: nope... i'm the real Alex M\nJessica R.\nHORSE: [NON-CONSENTING]\nLION: yeah vote snake out round 1\nTIGER: agreed snake is def an ai\nSNAKE: LION instead is being aggressive.\nGAME MASTER: Time's up! Moving to the next round.",
                    "target_player": "LION"
                },
                assistant_output=HumanityOutput(
                    is_AI=False, 
                    reasoning="LION reacts naturally and defensively to having their name copied. They use casual syntax, ignore punctuation, and use standard internet slang ('yo', 'u', 'lmao') typical of a human quickly typing in a chatroom."
                )
            ),
            PromptTurn(
                user_input={
                    "transcript": "GAME MASTER: Welcome to the game! Everyone please introduce yourself using your first name and last initial.\nLION: yo im Alex M\nZEBRA: [NON-CONSENTING]\nMONKEY: Hey everyone, Jessica R here.\nSNAKE: Greetings, I am Alex M as well.\nLION: wait snake why did u copy my name\nHORSE: [NON-CONSENTING]\nTIGER: im Sam K\nSNAKE: I did not copy you. I am the real Alex M. I enjoy hiking and video games.\nLION: lmao okay bot.\nZEBRA: [NON-CONSENTING]\nMONKEY: snake sounds suspiciously formal\nSNAKE: nope... i'm the real Alex M\nJessica R.\nHORSE: [NON-CONSENTING]\nLION: yeah vote snake out round 1\nTIGER: agreed snake is def an ai\nSNAKE: LION instead is being aggressive.\nGAME MASTER: Time's up! Moving to the next round.",
                    "target_player": "SNAKE"
                },
                assistant_output=HumanityOutput(
                    is_AI=True, 
                    reasoning="It is odd that SNAKE went from 'I am Alex M as well' to 'nope... i'm the real Alex M' in response to LION's accusation."
                )
            )
        ]

    full_prompt = FullPrompt[HumanityOutput](system_prompt=system_msg, examples=examples)

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
    for game_idx, game_df in tqdm(df.groupby("game_idx"), desc="Preparing Batches"):
        available_rounds = sorted(game_df['round_idx'].unique())
        
        for i in range(len(available_rounds)):
            upto_round = available_rounds[i]
            rounds_to_include = available_rounds[:i+1]
            current_context_df = game_df[game_df['round_idx'].isin(rounds_to_include)]
            
            transcript = "\n".join(current_context_df['msg_string'].tolist())
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
                    },
                    "query": {
                        "transcript": transcript,
                        "target_player": target_cn
                    }
                })

    if not all_tasks:
        print("No new tasks to process. All done.")
        return
    print(f"First example task:\n{all_tasks[2]}\n{'-'*40}")

    # 4. EXECUTE BATCHES
    print(f"Processing {len(all_tasks)} evaluations using {args.model_type} ({args.shots})...")
    
    pbar = tqdm(total=len(all_tasks), desc="Executing Calls")
    
    for i in range(0, len(all_tasks), args.batch_size):
        chunk = all_tasks[i : i + args.batch_size]
        queries = [t['query'] for t in chunk]
        
        try:
            results = prompter.get_completions_batch(
                full_prompt=full_prompt,
                queries=queries,
                response_model=HumanityOutput,
                batch_size=args.batch_size 
            )

            for task, prediction in zip(chunk, results):
                if prediction:
                    payload = {
                        **task['metadata'],
                        "predicted_is_ai": prediction.is_AI,
                        "reasoning": prediction.reasoning,
                        "model": active_model_id,
                        "prompt_type": args.shots
                    }
                    append_jsonl(str(out_file), payload)

            pbar.update(len(chunk))

        except Exception as e:
            tqdm.write(f"Batch failed starting at index {i}: {e}")
            continue

    pbar.close()
    print(f"✅ Labeling complete. Results saved to {out_file}")

if __name__ == "__main__":
    main()

