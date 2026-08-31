'''
python ./src/analysis/paird_t_test.py
'''
import pandas as pd
from scipy.stats import ttest_rel

DATA_PATH = "data/flattened_votes.tsv"

def main():
    data = pd.read_csv(DATA_PATH, sep="\t")

    # Convert outcome to numeric
    data["voted_for_is_ai"] = data["voted_for_is_ai"].astype(int)

    # Encode day
    data["day"] = data["date"].apply(lambda x: "day2" if x != "6-09" else "day1")

    # Drop missing values
    data = data.dropna(subset=["voted_for_is_ai", "voter_id", "day"])

    # Compute per-student detection accuracy per day
    acc_by_student_day = (
        data
        .groupby(["voter_id", "day"])["voted_for_is_ai"]
        .mean()
        .reset_index()
    )

    # Pivot to wide format: one row per student
    acc_wide = acc_by_student_day.pivot(
        index="voter_id",
        columns="day",
        values="voted_for_is_ai"
    )

    # Keep only students with both Day 1 and Day 2 data
    acc_wide = acc_wide.dropna(subset=["day1", "day2"])

    # Paired t-test (within-participant comparison)
    t_stat, p_val = ttest_rel(acc_wide["day1"], acc_wide["day2"])

    print(f"Paired t-test results:")
    print(f"t = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"Mean Day 1 accuracy: {acc_wide['day1'].mean():.3f}")
    print(f"Mean Day 2 accuracy: {acc_wide['day2'].mean():.3f}")

if __name__ == "__main__":
    main()