'''
python ./src/analysis/regression.py
'''
import random
import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = "data/flattened_votes.tsv"

def main():
    data = pd.read_csv(DATA_PATH, sep="\t")
    
    # 1. Convert boolean target and date
    data["voted_for_is_ai"] = data["voted_for_is_ai"].astype(int)
    data["second_day"] = (data["date"] != "6-09").astype(int)
    data["pr_centered"] = data["players_remaining"] - data["players_remaining"].mean()
    # data["round_idx"] = data["round_idx"] - data["round_idx"].mean()
    
    # 2. Clean NaNs
    data = data.dropna(subset=["voted_for_is_ai", "round_idx", "second_day", "voter_id", "pr_centered", "games_played"])

    # 3. Fit the Mixed Effects Model
    # Fixed effects: round_idx, second_day, players_remaining
    # Random effect: Intercept grouped by voter_id
    # model = smf.mixedlm(
    #     # "voted_for_is_ai ~ games_played + round_idx * second_day *  pr_centered - round_idx:second_day:pr_centered", # this is the best one...
    #     "voted_for_is_ai ~ games_played + round_idx * second_day *  pr_centered - round_idx:second_day:pr_centered", # this is the best one...

    #     data, 
    #     groups=[data["game_id"], data["voter_id"]]
    # )
    model = smf.mixedlm(
        "voted_for_is_ai ~ games_played + round_idx * second_day * pr_centered - round_idx:second_day:pr_centered",
        data,
        groups=data["game_id"],  
        vc_formula={"voter_id": "0 + C(voter_id)"} 
    )
    result = model.fit(method='cg')
    print(result.summary())

if __name__=="__main__":

    main()
