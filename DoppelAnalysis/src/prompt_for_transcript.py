"""
python ./src/prompt_for_transcript.py \
  --model_type gpt \
  --batch_size 10
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, Field

# Internal imports
from utils.prompters import OpenAIPrompter, QwenPrompter, LlamaPrompter
from utils.prompt_objects import FullPrompt
from utils.file_io import load_jsonl, append_jsonl


# ───────────────── CONFIG ─────────────────
DATA_PATH = Path("./data/cleaned_transcripts.jsonl")
OUTPUT_PATH = Path("./data/annotations/transcript_code_labels.jsonl")   
CONTEXT_SIZE = 11          # total lines (target + context)
HALF_WINDOW = 5
RESUME = True


# ───────────────── OUTPUT SCHEMA ─────────────────
class CodeAnnotation(BaseModel):
    perceptions_of_ai: bool = Field(
        description="Beliefs or assumptions about what AI is, how it works, or its capabilities."
    )
    ai_detection_heuristics: bool = Field(
        description="Strategies or cues used to identify AI vs humans."
    )
    learning_adaptation: bool = Field(
        description="Evidence of learning, adaptation, or strategy change over time."
    )
    social_emotional_responses: bool = Field(
        description="Emotions, humor, fear, enjoyment, or social dynamics."
    )
    privacy_risk: bool = Field(
        description="Concerns about data privacy, inference, surveillance, or identity risk."
    )
    design_feedback: bool = Field(
        description="Suggestions or critiques for improving AI or the system."
    )


# ───────────────── ARGS ─────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Annotate transcript utterances with qualitative codes.")
    parser.add_argument("--model_type", required=True, choices=["gpt", "qwen", "llama"])
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--debug", type=int, default=0)
    return parser.parse_args()


# ───────────────── PROMPTER ─────────────────
def get_prompter(model_type, model_id, debug):
    if model_type == "gpt":
        return OpenAIPrompter(model_id=model_id or "gpt-4o-mini"), model_id or "gpt-4o-mini"
    if model_type == "qwen":
        return QwenPrompter(model_id=model_id or "Qwen/Qwen3.5-27B", debug=bool(debug)), model_id
    if model_type == "llama":
        return LlamaPrompter(model_id=model_id or "meta-llama/Llama-3.1-8B-Instruct", debug=bool(debug)), model_id
    raise ValueError(f"Unknown model type: {model_type}")


# ───────────────── CONTEXT WINDOW ─────────────────
def build_context(entries, idx):
    """
    Returns an 11-line context window centered on idx when possible.
    """
    n = len(entries)

    start = max(0, idx - HALF_WINDOW)
    end = min(n, idx + HALF_WINDOW + 1)

    # Expand window if we're near edges
    while (end - start) < CONTEXT_SIZE:
        if start > 0:
            start -= 1
        elif end < n:
            end += 1
        else:
            break

    window = entries[start:end]

    return "\n".join(
        f"{e['speaker']}: {e['quote']}" for e in window if e.get("quote")
    )


# ───────────────── MAIN ─────────────────
def main():
    args = parse_args()
    os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)

    prompter, active_model = get_prompter(args.model_type, args.model_id, args.debug)

    system_prompt = (
        "You are a trained qualitative research annotator.\n\n"
        "You will be given:\n"
        "1) A short transcript excerpt (with surrounding context)\n"
        "2) A target utterance marked clearly\n\n"
        "Your task is to label only the TARGET utterance.\n"
        "Do not infer beyond what is said.\n"
        "Return boolean values only."
    )

    full_prompt = FullPrompt[CodeAnnotation](
        system_prompt=system_prompt,
        examples=[]
    )

    # Load transcript
    entries = load_jsonl(DATA_PATH)
    entries = entries[:10]

    # Resume support
    processed_uids = set()
    if RESUME and Path(OUTPUT_PATH).exists():
        existing = load_jsonl(OUTPUT_PATH)
        processed_uids = {e["uid"] for e in existing}
        print(f"Resuming: {len(processed_uids)} already annotated.")

    tasks = []

    for idx, entry in enumerate(entries):
        speaker = entry.get("speaker", "")
        uid = entry.get("uid")

        # Skip reviewers for labeling (but NOT context)
        if "Reviewer" in speaker:
            continue

        if uid in processed_uids:
            continue

        context = build_context(entries, idx)

        query = {
            "context": context,
            "target_utterance": f"{speaker}: {entry.get('quote', '')}"
        }

        tasks.append({
            "uid": uid,
            "speaker": speaker,
            "quote": entry.get("quote"),
            "query": query
        })

    if not tasks:
        print("No new utterances to annotate.")
        return

    print(f"Annotating {len(tasks)} utterances using {active_model}")
    pbar = tqdm(total=len(tasks))

    for i in range(0, len(tasks), args.batch_size):
        chunk = tasks[i:i + args.batch_size]
        queries = [t["query"] for t in chunk]

        try:
            results = prompter.get_completions_batch(
                full_prompt=full_prompt,
                queries=queries,
                response_model=CodeAnnotation,
                batch_size=args.batch_size
            )

            for task, annotation in zip(chunk, results):
                payload = {
                    "uid": task["uid"],
                    "speaker": task["speaker"],
                    "quote": task["quote"],
                    "labels": annotation.model_dump(),
                    "model": active_model
                }
                append_jsonl(OUTPUT_PATH, payload)

            pbar.update(len(chunk))

        except Exception as e:
            tqdm.write(f"Batch failed at index {i}: {e}")
            continue

    pbar.close()
    print(f"✅ Annotation complete → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()