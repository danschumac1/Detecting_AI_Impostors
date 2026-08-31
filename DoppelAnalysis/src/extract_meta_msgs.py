'''
2026-05-01
How to run:
   python ./src/extract_meta_msgs.py
'''
import pandas as pd

def main():
    df = pd.read_json("./data/apply_chat_codes_adjudicated.jsonl", lines=True)
    print(df["labels"].value_counts())
        
    df_meta = df[df["labels"].apply(
        lambda x: isinstance(x, list) and "meta" in x
    )]

    print(len(df_meta))
    for msg in df_meta["content"]:
        print(msg)
        print("-" * 80)

if __name__ == "__main__":
    main()