import streamlit as st
import json
import random
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────
DATA_PATH = Path("./data/streamlit/rounds_based.jsonl")
ANNOTATION_DIR = Path("./data/streamlit/annotators/")
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_SIZE = 50

# ─── Helper Functions ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with DATA_PATH.open("r", encoding="utf-8") as fi:
        all_data = [json.loads(line) for line in fi]
    random.seed(SEED)
    return random.sample(all_data, SAMPLE_SIZE)

def get_annotation_path(annotator):
    return ANNOTATION_DIR / f"guess_the_ai_{annotator}.jsonl"

def load_annotations(annotator):
    path = get_annotation_path(annotator)
    if path.exists():
        with path.open("r", encoding="utf-8") as fi:
            return {json.loads(line)["key"]: json.loads(line) for line in fi}
    return {}

def save_annotation(annotator, annotation):
    path = get_annotation_path(annotator)
    with path.open("a", encoding="utf-8") as fo:
        json.dump(annotation, fo)
        fo.write("\n")
    print(f"SAVING TO: {path}")


# ─── Streamlit App ─────────────────────────────────────────────────────────
def main():
    st.title("🔍 GUESS THE AI")

    # Annotator identity
    annotator_name = st.sidebar.selectbox("Annotator", ["annot1", "annot2"])

    data = load_data()
    annotations = load_annotations(annotator_name)

    # Progress
    annotated_keys = {
        k for k, v in annotations.items()
        if v.get("selected_ai")
    }
    progress = sum(1 for item in data if item["key"] in annotated_keys)
    st.sidebar.markdown(f"**Progress:** {progress} / {len(data)} rounds completed ✅")

    # Round selection
    idx = st.sidebar.number_input("Round Index", 0, len(data) - 1, step=1)
    item = data[idx]
    key = item["key"]
    st.markdown(f"**Round {idx + 1} / {len(data)}**  &nbsp;|&nbsp;  **Key:** `{key}`")

    # Dialogue display
    st.markdown("### 💬 Dialogue")
    for line in item["dialog"].split("\n"):
        if ":" in line:
            speaker, text = line.split(":", 1)
            st.markdown(f"**{speaker.strip()}**: {text.strip()}")
    st.markdown("---")

    # AI guessing
    code_names = list(item["identities"].keys())
    default_ai = annotations.get(key, {}).get("selected_ai", [])
    selected_ai = st.multiselect(
        "❓ Which codenames do you believe to be AI?",
        code_names,
        default=default_ai
    )

    # Save
    if st.button("💾 Save Annotation"):
        if selected_ai:
            annotation = {
                "key": key,
                "selected_ai": selected_ai,
                "date": item["date"],
                "identities": item["identities"],
                "dialog": item["dialog"],
                "annotator": annotator_name
            }
            save_annotation(annotator_name, annotation)
            st.success("✅ Annotation saved!")
        else:
            st.warning("⚠️ Please select at least one codename before saving.")

    # Already annotated?
    if key in annotations:
        st.info("✅ This round has already been annotated by you.")
        st.json(annotations[key])

if __name__ == "__main__":
    main()
