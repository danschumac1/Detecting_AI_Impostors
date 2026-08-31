"""
2026-04-29

Run with:
    streamlit run src/app_transcript_labeling.py
"""

import json
from pathlib import Path
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DATA_PATH = Path("data/cleaned_transcripts.jsonl")
ANNOTATION_DIR = Path("data/streamlit/annotators")
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_SIZE = 11
HALF_WINDOW = 5

ANNOTATORS = ["annot1", "annot2", "annot3"]

CODES = {
    "🧠 Perceptions of AI": "perceptions_of_ai",
    "🕵️ AI Detection Heuristics": "ai_detection_heuristics",
    "📈 Learning & Adaptation": "learning_adaptation",
    "😄 Social / Emotional Responses": "social_emotional_responses",
    "🔐 Privacy & Risk": "privacy_risk",
    "🛠️ Design Feedback": "design_feedback",
}

NONE_KEY = "none_of_the_above"
EXCLUDE_SPEAKERS = ["Researcher", "Reviewer"]

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_transcripts():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_context(entries, idx):
    n = len(entries)
    start = max(0, idx - HALF_WINDOW)
    end = min(n, idx + HALF_WINDOW + 1)

    while (end - start) < CONTEXT_SIZE:
        if start > 0:
            start -= 1
        elif end < n:
            end += 1
        else:
            break

    return entries[start:end], start

def annotation_path(annotator):
    return ANNOTATION_DIR / f"transcripts_{annotator}.jsonl"

def load_annotations(annotator):
    path = annotation_path(annotator)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return {json.loads(line)["uid"]: json.loads(line) for line in f}

def save_annotation(annotator, payload):
    path = annotation_path(annotator)
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")

def clear_widget_state(uid):
    for key in list(st.session_state.keys()):
        if key.startswith(f"{uid}_"):
            del st.session_state[key]

# ─────────────────────────────────────────────────────────────
# Landing Screen
# ─────────────────────────────────────────────────────────────
def landing_screen():
    st.title("📝 Transcript Coding Interface")

    
    st.markdown(
    """
## Annotator Guidelines

### General Instructions for Annotators

- Apply **all relevant codes** that substantively apply. Codes are **not mutually exclusive**.
- If **no codes apply**, select **None of the Above**.
- Base decisions on **what the students say** and the **context of the conversation**.
- Apply codes **only to the highlighted utterance**, not the surrounding dialogue.

---

## 1. 🧠 Perceptions of AI

**Definition**  
Participants’ beliefs, assumptions, or mental models about what AI is, how it works, what it knows, or its capabilities and limitations.

**Include When:**
- A participant explains or defines AI
- A participant evaluates AI’s intelligence, competence, or realism
- A participant compares AI to humans (e.g., smarter, worse, limited)
- A participant reflects on whether AI can replace humans or improve over time

**Examples:**
- “AI isn’t as smart as I thought.”
- “Robots probably can’t replace us.”
- “It uses numbers to analyze.”
- “It just types what it is taught to.”

---

## 2. 🕵️ AI Detection Heuristics

**Definition**  
Strategies, cues, or reasoning participants use to distinguish AI from humans during interaction.

**Include When:**
- A participant identifies linguistic, behavioral, or stylistic clues
- A participant explains why someone or something “felt like AI”
- A participant discusses how AI–human differences shaped their voting behavior

**Examples:**
- “It kept repeating itself.”
- “Nobody can type that fast.”
- “Bots always punctuate everything.”
- “It was the first one to answer every time.”

---

## 3. 📈 Learning & Adaptation

**Definition**  
Evidence that participants reflect on experience, improve strategies, or adapt behavior across rounds or interactions.

**Include When:**
- The participant describes getting better at detecting AI
- The participant references past rounds or learning patterns
- The participant intentionally changes behavior to test or trick AI
- The participant reasons about how the AI is trained or updated

**Exclude When:**
- The statement is only a general belief about AI with no change over time  
  (code as *Perceptions of AI* instead)

**Examples:**
- “After the first round, I started watching how fast they replied.”
- “I learned not to trust the first answer anymore.”
- “We changed how we talked to see if it would mess up.”

---

## 4. 😄 Social / Emotional Responses

**Definition**  
Emotional reactions, affective states, and social dynamics related to interacting with AI or other players.

**Include When:**
- Participants express enjoyment, humor, fear, excitement, frustration, or boredom
- Participants comment on group dynamics or social interaction quality
- Laughter, jokes, or playfulness tied to AI interaction

**Examples:**
- “It was really fun.”
- “I got scared.”
- “That was insane.”
- “I couldn’t stop laughing.”

---

## 5. 🔐 Privacy & Risk

**Definition**  
Participant concerns, beliefs, or attitudes about data collection, storage, inference, misuse, surveillance, or personal risk related to AI.

**Include When:**
- Discussion of personal data being stored, leaked, inferred, or sold
- Expressions of comfort or discomfort with data retention
- Mentions of identity theft, impersonation, or scams

**Examples:**
- “It leaked my information.”
- “They could steal my identity.”
- “If companies already have my info…”

---

## 6. 🛠️ Design Feedback

**Definition**  
Explicit suggestions, critiques, or recommendations for improving AI behavior, system design, interface, or gameplay mechanics.

**Include When:**
- Suggestions to make AI more human‑like or less detectable
- Proposals for new features, constraints, or behaviors
- Critiques of system performance or design

**Examples:**
- “It should look more like phone messages.”
- "I wish the rounds were longer.”

---

### Notes on Multi‑Coding

A single utterance may receive **multiple codes**, especially when a participant:
- Describes a detection cue **and** evaluates AI intelligence
- Expresses emotion while discussing privacy or inference
- Reflects on learning **and** proposes design improvements

Annotators should document ambiguous cases for discussion during reliability meetings.
"""
    )
    

    annotator = st.selectbox("Select annotator ID", ANNOTATORS)

    if st.button("🚀 Start Annotating"):
        st.session_state.started = True
        st.session_state.annotator = annotator
        st.session_state.idx = 0
        st.rerun()

# ─────────────────────────────────────────────────────────────
# Annotation Screen
# ─────────────────────────────────────────────────────────────
def annotation_screen():
    annotator = st.session_state.annotator

    transcripts = load_transcripts()
    annotations = load_annotations(annotator)

    labelable = [
        (i, e) for i, e in enumerate(transcripts)
        if not any(x in e.get("speaker", "") for x in EXCLUDE_SPEAKERS)
    ]

    total = len(labelable)
    idx = st.session_state.idx
    idx = max(0, min(idx, total - 1))
    st.session_state.idx = idx

    real_idx, entry = labelable[idx]
    uid = entry["uid"]

    existing = annotations.get(uid, {}).get("labels", {})

    # ─── Sidebar ──────────────────────────────────────────────
    st.sidebar.markdown(f"**Annotator:** `{annotator}`")
    st.sidebar.markdown(f"**Progress:** {len(annotations)} / {total}")
        
    if st.sidebar.button("🏠 Return to Guidelines"):
        st.session_state.started = False
        st.session_state.idx = 0
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
### Coding Reminders

**🧠 Perceptions of AI**  
Beliefs about what AI is, how it works, or what it can do.

**🕵️ AI Detection Heuristics**  
Clues or strategies used to tell AI from humans.

**📈 Learning & Adaptation**  
Changes in strategy, learning across rounds, or testing AI.

**😄 Social / Emotional Responses**  
Emotions, humor, fear, enjoyment, or group dynamics.

**🔐 Privacy & Risk**  
Concerns about data, surveillance, misuse, or identity risk.

**🛠️ Design Feedback**  
Suggestions or critiques for improving AI or the system.

—

✔ Codes are **not mutually exclusive**  
✔ Use context, label **target utterance only**  
✔ Select **None of the Above** if no codes apply  
""")

    st.title("📝 Transcript Coding Interface")
    st.markdown(f"**Utterance {idx + 1} / {total}** | UID `{uid}`")

    col_text, col_codes = st.columns([3, 1])

    # ─── Column A: Context ────────────────────────────────────
    with col_text:
        st.markdown("### 💬 Dialogue Context")
        context, start = build_context(transcripts, real_idx)

        for j, e in enumerate(context):
            speaker = e.get("speaker", "")
            quote = e.get("quote", "")
            is_target = (start + j == real_idx)

            if is_target:
                st.markdown(f"👉 **🟨 {speaker}: {quote}**")
            else:
                st.markdown(f"**{speaker}:** {quote}")

    # ─── Column B: Codes ──────────────────────────────────────
    with col_codes:
        st.markdown("### 🏷 Codes")

        label_values = {}
        any_code_checked = False

        for label, key in CODES.items():
            checked = st.checkbox(
                label,
                value=existing.get(key, False),
                key=f"{uid}_{key}"
            )
            label_values[key] = checked
            any_code_checked |= checked

        none_checked = st.checkbox(
            "None of the Above",
            value=existing.get(NONE_KEY, False),
            key=f"{uid}_{NONE_KEY}"
        )

        # Mutual exclusion logic
        if none_checked and any_code_checked:
            for key in CODES.values():
                st.session_state[f"{uid}_{key}"] = False
            any_code_checked = False

        if any_code_checked and none_checked:
            st.session_state[f"{uid}_{NONE_KEY}"] = False
            none_checked = False

        st.markdown("---")

        col_back, col_next = st.columns(2)

        with col_back:
            if st.button("⬅ Back") and idx > 0:
                clear_widget_state(uid)
                st.session_state.idx -= 1
                st.rerun()

        with col_next:
            if st.button("💾 Save and Next"):
                if not any_code_checked and not none_checked:
                    st.warning("Select at least one code or None of the Above.")
                else:
                    payload = {
                        "uid": uid,
                        "speaker": entry.get("speaker"),
                        "quote": entry.get("quote"),
                        "labels": {**label_values, NONE_KEY: none_checked},
                        "annotator": annotator,
                    }
                    save_annotation(annotator, payload)
                    clear_widget_state(uid)

                    if idx + 1 < total:
                        st.session_state.idx += 1
                        st.rerun()
                    else:
                        st.success("✅ All utterances annotated!")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    if "started" not in st.session_state:
        st.session_state.started = False

    if not st.session_state.started:
        landing_screen()
    else:
        annotation_screen()

if __name__ == "__main__":
    main()