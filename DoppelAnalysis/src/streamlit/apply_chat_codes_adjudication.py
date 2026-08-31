import random
import streamlit as st
import json
from pathlib import Path

# ─── Paths and Config ─────────────────────────────────────────────────────
ROUND_BUCKETS_PATH = Path("./data/streamlit/round_buckets.json")
ANNOTATOR_DIR = Path("./data/streamlit/annotators/")
ANNOTATOR_DIR.mkdir(parents=True, exist_ok=True)

ANNOT1_PATH = ANNOTATOR_DIR / "acc_annot1_cleaned.jsonl"
ANNOT2_PATH = ANNOTATOR_DIR / "acc_annot2_cleaned.jsonl"
ADJUDICATED_PATH = ANNOTATOR_DIR / "apply_chat_codes_adjudicated.jsonl"

LABELS = {
    "rand": "Irrelevant, chaotic, inappropriate, or misleading messages.",
    "folw": "Aligns with the majority without original input.",
    "crvt": "Organizes group action to vote for a player.",
    "obs": "Shares observations about another player's behavior.",
    "meta": "References human-only knowledge or exploits AI weaknesses.",
    "iceb": "Answers or follows up on the ice breaker question."
}

# ─── Loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_round_buckets():
    with ROUND_BUCKETS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def get_sampled_keys(data):
    keys = list(data.keys())
    random.seed(42)
    return random.sample(keys, 50)

@st.cache_data
def load_annotation_file(path):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return {json.loads(line)["msg_key"]: json.loads(line) for line in f}
    return {}

def save_adjudicated_annotations(annotations):
    with ADJUDICATED_PATH.open("a", encoding="utf-8") as f:
        for ann in annotations:
            json.dump(ann, f)
            f.write("\n")
    st.success(f"✅ Saved {len(annotations)} adjudicated messages.")

# ─── Streamlit App ────────────────────────────────────────────────────────
def main():
    st.title("⚖️ Adjudicate Chat Codes".upper())

    data = load_round_buckets()
    keys = get_sampled_keys(data)

    annot1_annotations = load_annotation_file(ANNOT1_PATH)
    annot2_annotations = load_annotation_file(ANNOT2_PATH)

    idx = st.sidebar.number_input("Round Index", 0, len(keys) - 1, step=1)
    round_key = keys[idx]
    round_messages = data[round_key]

    st.markdown(f"### Round: `{round_key}`")
    st.divider()

    collected_annotations = []

    for msg in round_messages:
        msg_key = f"{round_key}__{msg['message_idx']}"
        speaker = msg["code_name"]
        content = msg["message"]
        is_human = msg["human_bool"] == "True"

        col1, col2 = st.columns([4, 3])

        with col1:
            st.markdown(f"**{'🧍' if is_human else '🤖'} {speaker}:** {content}")

        if is_human:
            annot1_labels = annot1_annotations.get(msg_key, {}).get("labels", [])
            annot2_labels = annot2_annotations.get(msg_key, {}).get("labels", [])

            with col2:
                st.markdown(f"**annot1:** {', '.join(annot1_labels) or '∅'}")
                st.markdown(f"**annot2:** {', '.join(annot2_labels) or '∅'}")

                final_labels = st.multiselect(
                    label="Adjudicated Labels",
                    options=list(LABELS.keys()),
                    default=list(set(annot1_labels) & set(annot1_labels)),  # Start with agreement
                    format_func=lambda x: f"{x} - {LABELS[x]}",
                    key=f"adj_{msg_key}"
                )

                collected_annotations.append({
                    "msg_key": msg_key,
                    "round_key": round_key,
                    "msg_index": msg["message_idx"],
                    "code_name": msg["code_name"],
                    "content": msg["message"],
                    "labels": final_labels,
                    "date": msg["date"],
                    "annot1_labels": annot1_labels,
                    "annot2_labels": annot2_labels,
                    "annotator": "adjudicated"
                })

    if st.button("💾 Save Adjudicated Labels"):
        adjudicated = [ann for ann in collected_annotations if ann["labels"]]
        save_adjudicated_annotations(adjudicated)

if __name__ == "__main__":
    main()
