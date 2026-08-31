"""
python ./src/analysis/data_counts.py
"""

import pandas as pd


def avg_msg_len(df):
    """Average character length of messages, ignoring NaNs."""
    return df['message'].dropna().str.len().mean()


def avg_msgs_per_round(df):
    """Average number of messages per (game, round)."""
    return df.groupby(['game_idx', 'round_idx']).size().mean()


def avg_msgs_per_game(df):
    """Average number of messages per game."""
    return df.groupby('game_idx').size().mean()


def main():
    # Load dataset
    data = pd.read_csv('data/master_chat_log.csv')
    print(data.columns)

    print(data.head())

    # Remove GAME MASTER utterances
    data = data[data['code_name'] != 'GAME MASTER']
    data = data[data["message"].notna()]

    # --- Overall statistics ---
    print("\nOverall Statistics")
    print("-" * 30)

    print(f"Total utterances: {len(data)}")
    print(f"Average utterance length: {avg_msg_len(data):.2f}")
    # print(f"Average messages per round: {avg_msgs_per_round(data):.2f}")
    print(f"Average messages per game: {avg_msgs_per_game(data):.2f}")

    # --- AI vs Human breakdown ---
    ai_data = data[~data['human_bool']]
    human_data = data[data['human_bool']]

    print("\nAI Data Statistics")
    print("-" * 30)
    print(f"Total utterances: {len(ai_data)}")
    print(f"Average utterance length: {avg_msg_len(ai_data):.2f}")
    # print(f"Average messages per round: {avg_msgs_per_round(ai_data):.2f}")
    print(f"Average messages per game: {avg_msgs_per_game(ai_data):.2f}")

    print("\nHuman Data Statistics")
    print("-" * 30)
    print(f"Total utterances: {len(human_data)}")
    print(f"Average utterance length: {avg_msg_len(human_data):.2f}")
    # print(f"Average messages per round: {avg_msgs_per_round(human_data):.2f}")
    print(f"Average messages per game: {avg_msgs_per_game(human_data):.2f}")


if __name__ == "__main__":
    main()
