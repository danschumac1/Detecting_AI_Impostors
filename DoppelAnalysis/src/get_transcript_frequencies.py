import pandas as pd

ANNOT1_PATH = "./data/transcripts_annot1.jsonl"
ANNOT3_PATH = "./data/transcripts_annot3.jsonl"
ADJUD_PATH = "./data/transcript_adjudication.jsonl"

ID_COL = "uid"


def main():
    # -----------------------
    # Load files
    # -----------------------
    a1 = pd.read_json(ANNOT1_PATH, lines=True)
    a3 = pd.read_json(ANNOT3_PATH, lines=True)
    adj = pd.read_json(ADJUD_PATH, lines=True)

    # -----------------------
    # Normalize label columns
    # NOTE: different column names!
    # -----------------------
    a1_labels = pd.json_normalize(a1["labels"])
    a3_labels = pd.json_normalize(a3["labels"])
    adj_labels = pd.json_normalize(adj["final_labels"])

    a1 = pd.concat([a1[[ID_COL]], a1_labels], axis=1)
    a3 = pd.concat([a3[[ID_COL]], a3_labels], axis=1)
    adj = pd.concat([adj[[ID_COL]], adj_labels], axis=1)

    label_cols = a1_labels.columns

    # -----------------------
    # Find agreement rows
    # -----------------------
    merged = a1.merge(
        a3,
        on=ID_COL,
        suffixes=("_a1", "_a3"),
        how="inner"
    )

    agreement_mask = (
        merged[[f"{c}_a1" for c in label_cols]].values ==
        merged[[f"{c}_a3" for c in label_cols]].values
    ).all(axis=1)

    agreed = merged.loc[
        agreement_mask,
        [ID_COL] + [f"{c}_a1" for c in label_cols]
    ]

    agreed.columns = [ID_COL] + list(label_cols)

    # -----------------------
    # Combine agreed + adjudicated
    # -----------------------
    final_df = pd.concat([agreed, adj], ignore_index=True)

    # -----------------------
    # Stats
    # -----------------------
    print(f"Agreed rows: {len(agreed)}")
    print(f"Adjudicated rows: {len(adj)}")
    print(f"Total final rows: {len(final_df)}")

    category_frequencies = final_df[label_cols].sum().sort_values(ascending=False)

    print("\nCategory frequencies:")
    print(category_frequencies)

    # -----------------------
    # Safety checks
    # -----------------------
    assert final_df[ID_COL].is_unique, "Duplicate UIDs detected!"
    assert final_df[label_cols].isna().sum().sum() == 0, "Missing labels found!"


if __name__ == "__main__":
    main()