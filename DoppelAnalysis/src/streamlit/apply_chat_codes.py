import random
import streamlit as st
import json
from pathlib import Path

# ─── Paths and Config ───────────────────────────────────────────────
ROUND_BUCKETS_PATH = Path("./data/streamlit/round_buckets.json")
ANNOTATION_DIR = Path("./data/streamlit/annotators/")
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "rand": "Irrelevant, chaotic, inappropriate, or misleading messages.",
    "folw": "Aligns with the majority without original input.",
    "crvt": "Organizes group action to vote for a player.",
    "obs": "Shares observations about another player's behavior.",
    "meta": "References human-only knowledge or exploits AI weaknesses.",
    "iceb": "Answers or follows up on the ice breaker question."
}

# ─── Loaders ─────────────────────────────────────────────────────────
@st.cache_data
def load_round_buckets():
    with ROUND_BUCKETS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def get_sampled_keys(data):
    keys = list(data.keys())
    random.seed(42)
    return random.sample(keys, 50)

def get_annotation_path(annotator_name):
    return ANNOTATION_DIR / f"apply_chat_codes_{annotator_name}.jsonl"

def load_annotations(annotator_name):
    path = get_annotation_path(annotator_name)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return {json.loads(line)["msg_key"]: json.loads(line) for line in f}
    return {}

def save_annotation(annotator_name, annotations):
    path = get_annotation_path(annotator_name)
    with path.open("a", encoding="utf-8") as f:
        for ann in annotations:
            json.dump(ann, f)
            f.write("\n")
    print(f"SAVING TO: {path}")
    # print(f"Saved {len(annotations)} annotations to {path}")

# ─── Streamlit App ───────────────────────────────────────────────────
def main():
    st.title("🧠 APPLY CHAT CODES")

    st.sidebar.header("Annotator Setup")
    annotator_name = st.sidebar.selectbox("Select your name", options=["annot1", "annot2"])

    data = load_round_buckets()
    keys = get_sampled_keys(data)

    idx = st.sidebar.number_input("Round Index", 0, len(keys) - 1, step=1)
    round_key = keys[idx]
    round_messages = data[round_key]
    existing_annotations = load_annotations(annotator_name)

    # Track progress
    annotated_keys = {
        ann["round_key"]
        for ann in existing_annotations.values()
        if ann["labels"]
    }
    progress = sum(1 for key in keys if key in annotated_keys)
    st.sidebar.markdown(f"**Progress:** {progress} / {len(keys)} rounds completed ✅")

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
            previous = existing_annotations.get(msg_key, {}).get("labels", [])
            with col2:
                selected = st.multiselect(
                    label="Labels",
                    options=list(LABELS.keys()),
                    default=previous,
                    format_func=lambda x: f"{x} - {LABELS[x]}",
                    key=f"select_{msg_key}"
                )
                collected_annotations.append({
                    "msg_key": msg_key,
                    "round_key": round_key,
                    "msg_index": msg["message_idx"],
                    "code_name": msg["code_name"],
                    "content": msg["message"],
                    "labels": selected,
                    "date": msg["date"],
                    "annotator": annotator_name
                })

    if st.button("💾 Save All Annotations for This Round"):
        filtered = [ann for ann in collected_annotations if ann["labels"]]
        save_annotation(annotator_name, filtered)
        st.success(f"✅ Saved {len(filtered)} annotated messages for round `{round_key}`")

if __name__ == "__main__":
    main()
