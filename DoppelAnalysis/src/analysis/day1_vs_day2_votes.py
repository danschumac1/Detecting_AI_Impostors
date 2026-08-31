'''
Description: Compares overnight improvement (D1 vs D2) against 
             intra-day learning (1st game vs subsequent games) 
             for repeat players.

python ./src/analysis/day1_vs_day2_votes.py
'''
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Load and Basic Stats
    voting_df = pd.read_csv("./data/flattened_votes.tsv", sep="\t")
    
    print("--- DATASET SUMMARY ---")
    print(f"Total votes: {len(voting_df)}")
    print(f"Unique voters: {voting_df['voter_id'].nunique()}")

    # 2. Calculate Global Day 1 vs Day 2 (The "Overnight" Gap)
    day1_mask = voting_df["date"] == '6-09'
    day2_mask = voting_df["date"] == '6-10'
    
    day1_acc = voting_df[day1_mask]["voted_for_is_ai"].mean()
    day2_acc = voting_df[day2_mask]["voted_for_is_ai"].mean()
    overnight_gap = day2_acc - day1_acc

    print("\n--- OVERNIGHT CHECK ---")
    print(f"Day 1 Accuracy: {day1_acc:.2%} ({day1_mask.sum()} votes)")
    print(f"Day 2 Accuracy: {day2_acc:.2%} ({day2_mask.sum()} votes)")
    print(f"Overnight Gap:  {overnight_gap:+.2%}")

    # 3. Calculate Intra-day Learning (Filtered for Repeaters)
    day1_df = voting_df[day1_mask].copy()
    
    # Identify voters who played AT LEAST 2 games on Day 1
    games_per_voter = day1_df.groupby("voter_id")["game_id"].nunique()
    repeaters = games_per_voter[games_per_voter >= 2].index
    repeater_df = day1_df[day1_df["voter_id"].isin(repeaters)].copy()

    # Label chronological progress for repeaters
    repeater_df["is_first_game"] = (
        repeater_df.groupby("voter_id")["game_id"].transform("min") == repeater_df["game_id"]
    )

    first_game_acc = repeater_df[repeater_df["is_first_game"]]["voted_for_is_ai"].mean()
    subsequent_acc = repeater_df[~repeater_df["is_first_game"]]["voted_for_is_ai"].mean()
    intra_day_gap = subsequent_acc - first_game_acc

    print("\n--- INTRA-DAY CHECK (Day 1 Repeaters Only) ---")
    print(f"Repeat Voters:  {len(repeaters)}")
    print(f"Game 1 Acc:     {first_game_acc:.2%}")
    print(f"Subsequent Acc: {subsequent_acc:.2%}")
    print(f"Intra-day Gap:  {intra_day_gap:+.2%}")

    # 4. Visualization
    gaps = {
        'Overnight Improvement\n(Day 1 to Day 2)': overnight_gap,
        'Intra-day Learning\n(Repeaters: 1st vs Subsq)': intra_day_gap
    }
    
    plt.figure(figsize=(9, 6))
    colors = ['#4a90e2' if v > 0 else '#e24a4a' for v in gaps.values()]
    bars = plt.bar(gaps.keys(), [v * 100 for v in gaps.values()], color=colors, width=0.6)
    
    # Add value labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (1 if yval > 0 else -3), 
                 f"{yval:+.1f}%", ha='center', fontweight='bold')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel("Accuracy Change (Percentage Points)")
    plt.title("Voter Improvement: Overnight vs. Intra-day", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    plt.savefig("improvement_comparison.png")
    print("\nResults plotted to improvement_comparison.png")

if __name__ == "__main__":
    main()