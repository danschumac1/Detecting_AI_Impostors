import streamlit as st
import json
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────
DATA_PATH = Path("./data/streamlit/rounds_based.jsonl")
ANNOTATOR_PATHS = {
    "annot1": Path("./data/streamlit/annotators/guess_the_ai_annot1.jsonl"),
    "annot2": Path("./data/streamlit/annotators/guess_the_ai_annot2.jsonl"),
}
ADJUDICATED_PATH = Path("./data/streamlit/annotators/guess_the_ai_adjudicated.jsonl")

# ─── Helpers ───────────────────────────────────────────────────────────────
@st.cache_data
def load_base_data():
    with DATA_PATH.open("r", encoding="utf-8") as fi:
        return {json.loads(line)["key"]: json.loads(line) for line in fi}

@st.cache_data
def load_annotations(path):
    if path.exists():
        with path.open("r", encoding="utf-8") as fi:
            return {json.loads(line)["key"]: json.loads(line) for line in fi}
    return {}

def load_adjudicated():
    if ADJUDICATED_PATH.exists():
        with ADJUDICATED_PATH.open("r", encoding="utf-8") as fi:
            return {json.loads(line)["key"]: json.loads(line) for line in fi}
    return {}

def save_adjudication(annotation):
    with ADJUDICATED_PATH.open("a", encoding="utf-8") as fo:
        json.dump(annotation, fo)
        fo.write("\n")
    st.success(f"✅ Verdict saved to: {ADJUDICATED_PATH.name}")

# ─── Streamlit App ─────────────────────────────────────────────────────────
def main():
    st.title("⚖️ Adjudicate: Guess the AI".upper())

    # Load data
    base_data = load_base_data()
    annot1_data = load_annotations(ANNOTATOR_PATHS["annot1"])
    annot2_data = load_annotations(ANNOTATOR_PATHS["annot2"])
    adjudicated = load_adjudicated()

    # Keys to adjudicate
    shared_keys = sorted(set(annot1_data) & set(annot2_data))
    already_done = set(adjudicated)
    remaining = [k for k in shared_keys if k not in already_done]

    if not remaining:
        st.info("🎉 All rounds have been adjudicated!")
        return

    idx = st.sidebar.number_input("Select Index", 0, len(remaining) - 1, step=1)
    key = remaining[idx]
    round_data = base_data[key]
    annot1_guess = annot1_data.get(key, {}).get("selected_ai", [])
    annot2_guess = annot2_data.get(key, {}).get("selected_ai", [])

    # Display context
    st.markdown(f"### 🔑 Key: `{key}` | 📅 {round_data['date']}")
    st.markdown("#### 💬 Conversation")
    for line in round_data["dialog"].split("\n"):
        if ":" in line:
            speaker, text = line.split(":", 1)
            st.markdown(f"**{speaker.strip()}**: {text.strip()}")

    # Show guesses
    st.markdown("#### 🧠 Annotator Guesses")
    st.write(f"**annot1's guess:** {annot1_guess}")
    st.write(f"**annot2's guess:** {annot2_guess}")

    # Final adjudication
    st.markdown("#### 🏁 Final Verdict")
    code_names = list(round_data["identities"].keys())
    options = ["NONE"] + code_names
    final_selection = st.multiselect("Who is AI (final call)?", options)

    # Validation
    if "NONE" in final_selection and len(final_selection) > 1:
        st.warning("⚠️ You selected 'NONE' along with other codenames. Only one is allowed. Please revise.")

    valid_selection = (
        (final_selection == ["NONE"]) or
        ("NONE" not in final_selection and len(final_selection) > 0)
    )

    if st.button("💾 Save Final Verdict"):
        if not valid_selection:
            st.warning("⚠️ Invalid selection. Choose either 'NONE' or one or more codenames.")
        else:
            selected_ai = [] if final_selection == ["NONE"] else final_selection

            adjudicated_annotation = {
                "key": key,
                "selected_ai": selected_ai,
                "date": round_data["date"],
                "identities": round_data["identities"],
                "dialog": round_data["dialog"],
                "annot1_guess": annot1_guess,
                "annot2_guess": annot2_guess,
                "adjudicator": "gold"
            }
            save_adjudication(adjudicated_annotation)

if __name__ == "__main__":
    main()
