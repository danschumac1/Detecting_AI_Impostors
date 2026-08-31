'''
How to run:
   python ./src/statistical_tests/basic_statisitics.py
'''

import pandas as pd
import csv
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
import os
from datetime import datetime
from utils.file_io import grab_pair_id, load_json

def main():
    # # Count the number of total games, day 1 games, day 2 days.
    with open("./data/master_chat_log.csv", encoding="utf-8") as f:
        chat = list(csv.DictReader(f))
    consent_idxs = load_json(Path("./resources/consent_idxs.json"))
    data = [line for line in chat if grab_pair_id(line['player_id']) in consent_idxs] 

    print(f"TOTAL MSGS: {len(data)}")
    df = pd.DataFrame(data)
    day1 = df[df["date"].isin(["6-09","06-09"])]
    day2 = df[df["date"].isin(["6-10","06-10"])]
    print(f"DAY 1 MSGS: {len(day1)}")
    print(f"DAY 2 MSGS: {len(day2)}")

    # REMOVING BLANK MESSAGES. 
    day1 = day1[day1["message"] != ""]
    day2 = day2[day2["message"] != ""]

    print(f"DAY 1 NON-BLANK: {len(day1)}")
    print(f"DAY 2 NON-BLANK: {len(day2)}")

    print(f"TOTAL UNIQUE GAMES: {df['game_idx'].nunique()}")
    print(f"DAY 1 UNIQUE GAMES: {day1['game_idx'].nunique()}")
    print(f"DAY 2 UNIQUE GAMES: {day2['game_idx'].nunique()}")
    print(f"NUMBER HUMAN PLAYERS: {len(consent_idxs)}")

    # Count the number of total msgs, day 1 msgs, day 2, msgs.
    # repeat above but with only consenting students
    d1 = datetime.strptime("2025-06-09", "%Y-%m-%d").date()
    d2 = datetime.strptime("2025-06-10", "%Y-%m-%d").date()

    n_1 = 0
    n_2 = 0
    parent = Path("./data/lobbies")
    for folder in parent.iterdir():
        if folder.is_dir():
            lobby_num = int(folder.name.split("_")[1])
            if not 300 <= lobby_num < 900:
                continue

            time_path = folder / "starttime.txt"
            if time_path.exists():
                content = json.loads(time_path.read_text())
                num_rounds = len(content.keys()) 
                
                if num_rounds >= 3:
                    ts = datetime.fromisoformat(content["0"])
                    if ts.date() == d1:
                        n_1 += 1
                    elif ts.date() == d2:
                        n_2 += 1
                else:
                    continue

                
if __name__ == "__main__":
    main()