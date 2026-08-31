"""
Paired semantic similarity analysis with
Wilcoxon signed-rank tests and paired bootstrap confidence intervals.

How to run:
    CUDA_VISIBLE_DEVICES=3 python ./src/statistical_tests/paired_boot_wilcoxen.py
"""

# ───────────────────────── Imports & Setup ─────────────────────────
import sys
sys.path.append("src")

import os
# ───────────────────────── HuggingFace Cache ─────────────────────────
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface/transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")

import json
import csv
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer




# ───────────────────────── Constants ─────────────────────────
NON_LIWC_COLS = {
    "game_idx", "message_idx", "player_id_x", "code_name", "human_bool",
    "ColumnID", "Text", "Segment", "player_id", "days", "pair_idx",
    "date", "player_id_y", "message"
}


# ───────────────────────── Data Loading ─────────────────────────
def load_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ───────────────────────── Statistics ─────────────────────────
def paired_bootstrap_diff(x, y, n_boot=10_000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    assert len(x) == len(y) and len(x) > 0

    diffs = y - x
    obs_mean = diffs.mean()
    n = len(diffs)

    boot_means = np.array([
        diffs[rng.integers(0, n, n)].mean()
        for _ in range(n_boot)
    ])

    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    null_diffs = diffs - obs_mean
    null_boot = np.array([
        null_diffs[rng.integers(0, n, n)].mean()
        for _ in range(n_boot)
    ])

    p_value = np.mean(np.abs(null_boot) >= abs(obs_mean))
    return obs_mean, ci_lo, ci_hi, p_value


def report_paired_results(name, d1, d2):
    mean_diff, ci_lo, ci_hi, p_boot = paired_bootstrap_diff(d1, d2)
    w, p_wilc = wilcoxon(d1, d2)

    print(f"\n{name}")
    print(f"  Day 1 mean: {np.mean(d1):.3f}")
    print(f"  Day 2 mean: {np.mean(d2):.3f}")
    print(f"  Mean diff (Day2 − Day1): {mean_diff:.3f}")
    print(f"  95% bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Bootstrap p = {p_boot:.4f}")
    print(f"  Wilcoxon W = {w:.3f}, p = {p_wilc:.4f}")


# ───────────────────────── Pair Alignment ─────────────────────────
def align_pairs(d1: Dict, d2: Dict):
    keys = sorted(d1.keys() & d2.keys())
    return np.array([d1[k] for k in keys]), np.array([d2[k] for k in keys])


# ───────────────────────── Text Normalization ─────────────────────────
def clean_texts(texts, min_alpha_ratio=0.25):
    out = []
    for t in texts:
        if not t or not t.strip():
            continue
        ratio = sum(c.isalpha() for c in t) / max(len(t), 1)
        if ratio < min_alpha_ratio:
            continue
        t = re.sub(r"\s+", " ", t.strip())
        out.append(t)
    return out

# ───────────────────────── LIWC SCORE ─────────────────────────
def is_valid_player(pid):
    return pid and pid != "sys_1"


def extract_liwc_vector(row, liwc_keys):
    return [float(row[k]) if row[k] != "" else 0.0 for k in liwc_keys]


def clean_liwc(master_csv: str, liwc_csv: str):
    """
    Returns cleaned LIWC rows with metadata:
    [{date, player_id, pair_idx, liwc_vec}]
    """
    master = load_csv(master_csv)
    liwc = load_csv(liwc_csv)

    lookup = {
        (r["game_idx"], r["message_idx"]): (r["date"], r["player_id"])
        for r in master
    }

    liwc_keys = [k for k in liwc[0] if k not in NON_LIWC_COLS]
    out = []

    for row in liwc:
        if row["game_idx"] == "501":
            continue

        key = (row["game_idx"], row["message_idx"])
        if key not in lookup:
            continue

        date, pid = lookup[key]
        if not is_valid_player(pid):
            continue

        pair_idx = pid.split("_")[1] if "_" in pid else None
        out.append({
            "date": date,
            "player_id": pid,
            "pair_idx": pair_idx,
            "liwc_vec": extract_liwc_vector(row, liwc_keys)
        })

    return out


def build_liwc_vectors(liwc_rows, date):
    """
    Returns:
        { player_id : mean_liwc_vector } for one day
    """
    grouped = {}
    for r in liwc_rows:
        if r["date"] != date:
            continue
        grouped.setdefault(r["player_id"], []).append(r["liwc_vec"])

    return {
        pid: np.mean(vecs, axis=0)
        for pid, vecs in grouped.items()
    }


def liwc_pair_similarity(day_vecs):
    """
    day_vecs : {player_id: vector}

    Returns:
        { pair_id : cosine_similarity }
    """
    scores = {}
    pair_ids = {pid.split("_")[1] for pid in day_vecs if "_" in pid}

    for idx in pair_ids:
        hu = f"hu_{idx}"
        ai = f"ai_{idx}"
        if hu not in day_vecs or ai not in day_vecs:
            continue

        scores[idx] = cosine_similarity(
            [day_vecs[hu]],
            [day_vecs[ai]]
        )[0, 0]

    return scores

# ───────────────────────── BERTScore ─────────────────────────
def get_bertscorer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    scorer = BERTScorer(
        model_type="microsoft/deberta-large-mnli",
        lang="en",
        rescale_with_baseline=True,
        batch_size=8,
        device=device,
        use_fast_tokenizer=False
    )

    scorer._tokenizer.model_max_length = 512
    scorer._tokenizer.init_kwargs["truncation"] = True
    scorer._tokenizer.init_kwargs["max_length"] = 512

    return scorer


def bertscore_similarity(day_data, scorer):
    scores = {}
    for pair in day_data:
        pid = next(iter(pair))
        c = pair[pid]

        hu = clean_texts(c.get("hu", []))
        ai = clean_texts(c.get("ai", []))
        if not hu or not ai:
            continue

        n = min(len(hu), len(ai))
        with torch.no_grad():
            _, _, F1 = scorer.score(ai[:n], hu[:n])
        scores[pid] = F1.mean().item()

    return scores


# ───────────────────────── BERT Cosine Similarity ─────────────────────────
def bert_cosine_similarity(day_data, tokenizer, model):
    scores = {}
    for pair in day_data:
        pid = next(iter(pair))
        c = pair[pid]
        if not c["hu"] or not c["ai"]:
            continue

        hu = tokenizer(" ".join(c["hu"]), return_tensors="pt", truncation=True)
        ai = tokenizer(" ".join(c["ai"]), return_tensors="pt", truncation=True)

        with torch.no_grad():
            hu_vec = model(**hu).last_hidden_state.mean(dim=1)
            ai_vec = model(**ai).last_hidden_state.mean(dim=1)

        scores[pid] = cosine_similarity(
            hu_vec.cpu().numpy(),
            ai_vec.cpu().numpy()
        )[0, 0]

    return scores


# ───────────────────────── CountVectorizer ─────────────────────────
def cv_similarity(day1, day2):
    corpus, index = [], []

    for day_idx, day in enumerate([day1, day2], start=1):
        for pair in day:
            pid = next(iter(pair))
            c = pair[pid]
            if not c["hu"] or not c["ai"]:
                continue
            corpus.extend([" ".join(c["hu"]), " ".join(c["ai"])])
            index.append((day_idx, pid))

    X = normalize(CountVectorizer(max_features=10000).fit_transform(corpus))
    scores = {1: {}, 2: {}}

    for i in range(0, len(index), 2):
        d, pid = index[i]
        scores[d][pid] = cosine_similarity(X[i], X[i + 1])[0, 0]

    return scores[1], scores[2]


# ───────────────────────── Lexical Metrics ─────────────────────────
def lexical_metrics(day_data):
    out = {"bleu": {}, "meteor": {}, "rougeL": {}}
    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for pair in day_data:
        pid = next(iter(pair))
        c = pair[pid]
        if not c["hu"] or not c["ai"]:
            continue

        hu = " ".join(c["hu"])
        ai = " ".join(c["ai"])

        out["bleu"][pid] = sentence_bleu([hu.split()], ai.split(), smoothing_function=smoothie)
        out["meteor"][pid] = meteor_score([hu.split()], ai.split())
        out["rougeL"][pid] = rouge.score(hu, ai)["rougeL"].fmeasure

    return out


# ───────────────────────── Main Analysis ─────────────────────────
def main():
    day1 = load_jsonl(Path("./data/paired_indv_day1.jsonl"))
    day2 = load_jsonl(Path("./data/paired_indv_day2.jsonl"))

    print("\n=== Paired Semantic Similarity Analysis ===")

    
    # ── LIWC cosine similarity ─────────────────────────
    print("\n--- LIWC cosine similarity ---")

    consent_idxs = set(json.load(open("./resources/consent_idxs.json")))

    liwc_cleaned = clean_liwc(
        "./data/master_chat_log.csv",
        "./data/original/liwc.csv"
    )

    # consent filter
    liwc_cleaned = [
        r for r in liwc_cleaned
        if r["pair_idx"] is not None
        and int(r["pair_idx"]) in consent_idxs
    ]

    d1_liwc_vecs = build_liwc_vectors(liwc_cleaned, "6-09")
    d2_liwc_vecs = build_liwc_vectors(liwc_cleaned, "6-10")

    d1_liwc = liwc_pair_similarity(d1_liwc_vecs)
    d2_liwc = liwc_pair_similarity(d2_liwc_vecs)

    l1, l2 = align_pairs(d1_liwc, d2_liwc)
    report_paired_results("LIWC cosine similarity", l1, l2)
    # BERTScore
    scorer = get_bertscorer()
    b1, b2 = align_pairs(
        bertscore_similarity(day1, scorer),
        bertscore_similarity(day2, scorer)
    )
    report_paired_results("BERTScore F1", b1, b2)

    # BERT cosine
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    bc1, bc2 = align_pairs(
        bert_cosine_similarity(day1, tokenizer, model),
        bert_cosine_similarity(day2, tokenizer, model)
    )
    report_paired_results("BERT cosine similarity", bc1, bc2)

    # CountVectorizer
    cv1, cv2 = cv_similarity(day1, day2)
    c1, c2 = align_pairs(cv1, cv2)
    report_paired_results("CountVectorizer cosine similarity", c1, c2)

    # Lexical metrics
    lex1 = lexical_metrics(day1)
    lex2 = lexical_metrics(day2)
    for m in ["bleu", "meteor", "rougeL"]:
        l1, l2 = align_pairs(lex1[m], lex2[m])
        report_paired_results(m.upper(), l1, l2)



if __name__ == "__main__":
    main()

# """
# Paired semantic similarity analysis with
# Wilcoxon signed-rank tests and paired bootstrap CIs.

# How to run:
#     CUDA_VISIBLE_DEVICES=3 python ./src/statistical_tests/paired_boot_wilcoxen.py
# """

# # ───────────────────────── Imports & Setup ─────────────────────────
# import sys; sys.path.append("src")
# import os
# # Cache dirs

# os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
# print(os.environ["HF_HOME"])
# os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface/transformers")
# print(os.environ["TRANSFORMERS_CACHE"])
# os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")
# print(os.environ["HUGGINGFACE_HUB_CACHE"])

# from pathlib import Path

# from bert_score import BERTScorer

# import os
# import json
# import csv
# from pathlib import Path
# from typing import Dict, List, Tuple

# import numpy as np
# from scipy.stats import wilcoxon
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize

# import torch
# from transformers import BertTokenizer, BertModel

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer
# NON_LIWC_COLS = {
#     "game_idx", "message_idx", "player_id_x", "code_name", "human_bool", "ColumnID",
#     "Text", "Segment", "player_id", "days", "pair_idx", "date", "player_id_y", 'message'
# }

# import re

# def debug_tokenizer_probe(scorer, text, label, pair_id):
#     """
#     Attempts to tokenize a string safely and prints diagnostics.
#     """
#     try:
#         tok = scorer._tokenizer.encode(text)
#         print(f"[OK] {label} | pair={pair_id} | chars={len(text)} | tokens={len(tok)}")
#         return True
#     except Exception as e:
#         print("\n🚨 TOKENIZER FAILURE 🚨")
#         print(f"pair_id      = {pair_id}")
#         print(f"label        = {label}")
#         print(f"char_len     = {len(text)}")
#         print(f"text_preview = {repr(text[:200])}")
#         print(f"exception    = {type(e).__name__}: {e}")
#         print("🔴 STRING REJECTED 🔴\n")
#         return False


# import re

# def normalize_and_chunk_debug(texts, scorer, pair_id, role):
#     """
#     Aggressively logs every step of preprocessing.
#     """
#     chunks = []

#     print(f"\n=== Processing {role} text for pair {pair_id} ===")
#     print(f"Raw count: {len(texts)}")

#     for idx, t in enumerate(texts):
#         if not t or not t.strip():
#             print(f"[SKIP empty] {role}[{idx}]")
#             continue

#         t = t.strip()
#         char_len = len(t)

#         alpha_ratio = sum(c.isalpha() for c in t) / max(char_len, 1)

#         print(
#             f"{role}[{idx}] chars={char_len} "
#             f"alpha_ratio={alpha_ratio:.3f} "
#             f"preview={repr(t[:60])}"
#         )

#         # HARD REJECTION
#         if char_len > 2000:
#             print(f"❌ REJECTED: too long (>2000 chars)")
#             continue

#         if alpha_ratio < 0.25:
#             print(f"❌ REJECTED: low alpha ratio")
#             continue

#         t = re.sub(r"\s+", " ", t)

#         # Now probe tokenizer BEFORE accepting
#         ok = debug_tokenizer_probe(scorer, t, role, pair_id)
#         if not ok:
#             continue

#         chunks.append(t)

#     print(f"✅ Accepted {len(chunks)} chunks for {role}")
#     return chunks
# # ───────────────────────── Utility Functions ─────────────────────────
# def is_valid_player(pid):
#     return pid and pid != "sys_1"

# def load_jsonl(path: Path) -> List[dict]:
#     with path.open() as f:
#         return [json.loads(line) for line in f]

# def load_csv(path: str) -> List[dict]:
#     with open(path, newline="", encoding="utf-8") as f:
#         return list(csv.DictReader(f))

# def paired_bootstrap_diff(x, y, n_boot=10000, seed=42):
#     """
#     Paired bootstrap CI and two-sided bootstrap p-value for mean difference (y - x).
#     """
#     rng = np.random.default_rng(seed)
#     x = np.asarray(x)
#     y = np.asarray(y)

#     assert len(x) == len(y) and len(x) > 0

#     diffs = y - x
#     n = len(diffs)

#     obs_mean = diffs.mean()

#     # Bootstrap CI from observed paired differences
#     boot_means = np.array([
#         diffs[rng.integers(0, n, n)].mean()
#         for _ in range(n_boot)
#     ])

#     ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

#     # Null-centered bootstrap p-value
#     null_diffs = diffs - obs_mean

#     null_boot_means = np.array([
#         null_diffs[rng.integers(0, n, n)].mean()
#         for _ in range(n_boot)
#     ])

#     p_value = np.mean(np.abs(null_boot_means) >= abs(obs_mean))

#     #p_value = np.mean(null_boot_means >= obs_mean)


#     return obs_mean, ci_lo, ci_hi, p_value

# def report_paired_results(name, day1, day2):
#     mean_diff, ci_lo, ci_hi, p_value = paired_bootstrap_diff(day1, day2)
#     #w, p = wilcoxon(day1, day2, alternative='less')
#     w, p = wilcoxon(day1, day2)

#     print(f"\n{name}")
#     print(f"  Day 1 mean: {np.mean(day1):.3f}")
#     print(f"  Day 2 mean: {np.mean(day2):.3f}")
#     print(f"  Mean diff (Day2 − Day1): {mean_diff:.3f}")
#     print(f"  95% bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
#     print(f"  Bootstrap, p = {p_value:.4f}")
#     print(f"  Wilcoxon W = {w:.3f}, p = {p:.4f}")

# # ───────────────────────── Pair Alignment ─────────────────────────
# # Every metric returns a dict: {pair_id: score}

# def align_pairs(day1_dict, day2_dict):
#     keys = sorted(day1_dict.keys() & day2_dict.keys())
#     return (
#         np.array([day1_dict[k] for k in keys]),
#         np.array([day2_dict[k] for k in keys])
#     )
# # ───────────────────────── BERTScore Similarity ─────────────────────────

# def get_bertscorer():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     scorer = BERTScorer(
#         model_type="microsoft/deberta-large-mnli",
#         lang="en",
#         rescale_with_baseline=True,
#         device=device,
#         batch_size=8,
#         use_fast_tokenizer=False
#     )

#     # ✅ CRITICAL: freeze truncation once
#     scorer._tokenizer.model_max_length = 512
#     scorer._tokenizer.init_kwargs["truncation"] = True
#     scorer._tokenizer.init_kwargs["max_length"] = 512

#     return scorer


# def bertscore_similarity(day_data, scorer):
#     scores = {}

#     for pair in day_data:
#         pair_id = next(iter(pair))
#         content = pair[pair_id]

#         print(f"\n==============================")
#         print(f"PAIR {pair_id}")
#         print(f"==============================")

#         hu_chunks = normalize_and_chunk_debug(
#             content.get("hu", []), scorer, pair_id, "HU"
#         )
#         ai_chunks = normalize_and_chunk_debug(
#             content.get("ai", []), scorer, pair_id, "AI"
#         )

#         if not hu_chunks or not ai_chunks:
#             print("⚠️  No valid chunks — skipping pair")
#             continue

#         n = min(len(hu_chunks), len(ai_chunks))
#         refs = hu_chunks[:n]
#         cands = ai_chunks[:n]

#         print(f"🧮 Scoring {n} chunk pairs")

#         try:
#             with torch.no_grad():
#                 _, _, F1 = scorer.score(cands, refs)
#             score = F1.mean().item()
#             scores[pair_id] = score
#             print(f"✅ BERTScore F1 = {score:.4f}")

#         except Exception as e:
#             print("\n🔥 BERTScore CRASH 🔥")
#             print(f"pair_id  = {pair_id}")
#             print(f"chunks   = {n}")
#             print(f"exception= {type(e).__name__}: {e}")
#             raise  # re-raise so we see full trace

#     return scores# ───────────────────────── BERT Similarity ─────────────────────────

# def bert_similarity(day_data, tokenizer, model):
#     scores = {}
#     for pair in day_data:
#         pair_id = next(iter(pair))
#         content = pair[pair_id]

#         if not content["hu"] or not content["ai"]:
#             continue

#         hu_tok = tokenizer("\n".join(content["hu"]),
#                            return_tensors="pt",
#                            padding=True,
#                            truncation=True)
#         ai_tok = tokenizer("\n".join(content["ai"]),
#                            return_tensors="pt",
#                            padding=True,
#                            truncation=True)

#         with torch.no_grad():
#             hu_vec = model(**hu_tok).last_hidden_state.mean(dim=1)
#             ai_vec = model(**ai_tok).last_hidden_state.mean(dim=1)

#         sim = cosine_similarity(
#             hu_vec.cpu().numpy(),
#             ai_vec.cpu().numpy()
#         )[0, 0]

#         scores[pair_id] = sim

#     return scores

# # ───────────────────────── CountVectorizer Similarity ─────────────────────────

# def cv_similarity(day1, day2):
#     corpus = []
#     index = []

#     for day_idx, day in enumerate([day1, day2], start=1):
#         for pair in day:
#             pid = next(iter(pair))
#             c = pair[pid]
#             if not c["hu"] or not c["ai"]:
#                 continue
#             corpus.extend([" ".join(c["hu"]), " ".join(c["ai"])])
#             index.append((day_idx, pid))

#     X = CountVectorizer(max_features=10000).fit_transform(corpus)
#     X = normalize(X)

#     scores = {1: {}, 2: {}}
#     for i in range(0, len(index), 2):
#         day_idx, pid = index[i]
#         sim = cosine_similarity(X[i], X[i + 1])[0, 0]
#         scores[day_idx][pid] = sim

#     return scores[1], scores[2]

# # ───────────────────────── Lexical Metrics ─────────────────────────

# def extract_liwc_vector(row, liwc_keys):
#     return [float(row[k]) if row[k] != "" else 0.0 for k in liwc_keys]

# def clean_liwc(master_path: str, liwc_path: str) -> List[dict]:
#     liwc_data = load_csv(liwc_path)
#     master_data = load_csv(master_path)
#     lookup = {
#         (row["game_idx"], row["message_idx"]): (row["date"], row["player_id"])
#         for row in master_data
#     }
#     liwc_keys = [k for k in liwc_data[0].keys() if k not in NON_LIWC_COLS]
#     enriched = []
#     for row in liwc_data:
#         if row["game_idx"] == "501":
#             continue
#         key = (row["game_idx"], row["message_idx"])
#         if key not in lookup:
#             continue
#         date, pid = lookup[key]
#         if not is_valid_player(pid):
#             continue
#         pidx = pid.split("_")[1] if "_" in pid else None
#         enriched.append({"date": date, "player_id": pid, "pair_idx": pidx,
#                          "liwc_vec": extract_liwc_vector(row, liwc_keys)})
#     return enriched

# def build_liwc_vectors(liwc_rows, date):
#     """
#     Returns:
#         { player_id : avg_liwc_vector } for one day
#     """
#     out = {}
#     grouped = {}

#     for row in liwc_rows:
#         if row["date"] != date:
#             continue
#         pid = row["player_id"]
#         grouped.setdefault(pid, []).append(row["liwc_vec"])

#     for pid, vecs in grouped.items():
#         out[pid] = np.mean(np.array(vecs), axis=0)

#     return out

# def liwc_pair_similarity(day_vecs):
#     """
#     day_vecs: {player_id: liwc_vector}

#     Returns:
#         { pair_id : cosine_similarity }
#     """
#     scores = {}

#     pair_ids = {
#         pid.split("_")[1]
#         for pid in day_vecs
#         if "_" in pid
#     }

#     for idx in pair_ids:
#         hu = f"hu_{idx}"
#         ai = f"ai_{idx}"

#         if hu not in day_vecs or ai not in day_vecs:
#             continue

#         sim = cosine_similarity(
#             [day_vecs[hu]],
#             [day_vecs[ai]]
#         )[0, 0]

#         scores[idx] = sim

#     return scores




# def lexical_metrics(day_data):
#     out = {"bleu": {}, "meteor": {}, "rougeL": {}}

#     smoothie = SmoothingFunction().method4
#     rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

#     for pair in day_data:
#         pid = next(iter(pair))
#         c = pair[pid]
#         if not c["hu"] or not c["ai"]:
#             continue

#         hu = " ".join(c["hu"])
#         ai = " ".join(c["ai"])

#         out["bleu"][pid] = sentence_bleu(
#             [hu.split()], ai.split(), smoothing_function=smoothie
#         )
#         out["meteor"][pid] = meteor_score([hu.split()], ai.split())
#         out["rougeL"][pid] = rouge.score(hu, ai)["rougeL"].fmeasure

#     return out

# # ───────────────────────── Main Analysis ─────────────────────────

# def main():
#     day1 = load_jsonl(Path("./data/paired_indv_day1.jsonl"))
#     day2 = load_jsonl(Path("./data/paired_indv_day2.jsonl"))

#     print("\n=== Paired Semantic Similarity Analysis ===")

#     # # ── BERT ───────────────────────────
#     # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     # model = BertModel.from_pretrained("bert-base-uncased")

#     # d1_bert = bert_similarity(day1, tokenizer, model)
#     # d2_bert = bert_similarity(day2, tokenizer, model)

#     # b1, b2 = align_pairs(d1_bert, d2_bert)
#     # report_paired_results("BERT cosine similarity", b1, b2)

#     # # ── CountVectorizer ────────────────
#     # d1_cv, d2_cv = cv_similarity(day1, day2)
#     # c1, c2 = align_pairs(d1_cv, d2_cv)
#     # report_paired_results("CountVectorizer cosine similarity", c1, c2)

#     # print("\n--- LIWC cosine similarity ---")

#     # consent_idxs = set(json.load(open("./resources/consent_idxs.json")))
#     # liwc_rows = load_csv("./data/original/liwc.csv")
#     # master = load_csv("./data/master_chat_log.csv")

#     # # Rebuild cleaned LIWC rows (your logic, unchanged conceptually)
#     # liwc_cleaned = clean_liwc(
#     #     "./data/master_chat_log.csv",
#     #     "./data/original/liwc.csv"
#     # )

#     # # consent filter
#     # liwc_cleaned = [
#     #     r for r in liwc_cleaned
#     #     if int(r["pair_idx"]) in consent_idxs
#     # ]

#     # d1_vecs = build_liwc_vectors(liwc_cleaned, "6-09")
#     # d2_vecs = build_liwc_vectors(liwc_cleaned, "6-10")

#     # d1_liwc = liwc_pair_similarity(d1_vecs)
#     # d2_liwc = liwc_pair_similarity(d2_vecs)

#     # l1, l2 = align_pairs(d1_liwc, d2_liwc)

#     # report_paired_results("LIWC cosine similarity", l1, l2)

#     # # ── Lexical Metrics ────────────────
#     # lex1 = lexical_metrics(day1)
#     # lex2 = lexical_metrics(day2)

#     # for metric in ["bleu", "meteor", "rougeL"]:
#     #     m1, m2 = align_pairs(lex1[metric], lex2[metric])
#     #     report_paired_results(metric.upper(), m1, m2)

# # ── BERTScore ──────────────────────
#     scorer = get_bertscorer()

#     d1_bs = bertscore_similarity(day1, scorer)
#     d2_bs = bertscore_similarity(day2, scorer)

#     bs1, bs2 = align_pairs(d1_bs, d2_bs)
#     report_paired_results("BERTScore F1", bs1, bs2)
# if __name__ == "__main__":
#     main()
