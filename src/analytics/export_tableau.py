import pandas as pd

df = pd.read_csv("data/processed/contracts_features.csv")
tableau_cols = [
    "Awarding Agency", "Recipient Name", "Award Amount",
    "Contract Award Type", "contract_duration_days",
    "is_dod", "is_dc", "award_year", "is_q4",
    "recipient_experience", "agency_activity", "won_definitive"
]
df[tableau_cols].to_csv("data/output/project2_tableau_data.csv", index=False)
print("Saved data/output/project2_tableau_data.csv")
