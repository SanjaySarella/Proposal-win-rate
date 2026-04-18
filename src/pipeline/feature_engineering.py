import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/raw_contracts.csv")

print("Starting shape:", df.shape)

# Keep only Definitive Contracts and Delivery Orders
df = df[df["Contract Award Type"].isin(["DEFINITIVE CONTRACT", "DELIVERY ORDER"])]
print("After filter:", df.shape)

# Target variable — clean and defensible
df["won_definitive"] = (
    df["Contract Award Type"] == "DEFINITIVE CONTRACT"
).astype(int)

print("\nTarget distribution:")
print(df["won_definitive"].value_counts())
print(df["won_definitive"].value_counts(normalize=True).round(3))

# Clean award amount
df["Award Amount"] = pd.to_numeric(df["Award Amount"], errors="coerce")
df = df.dropna(subset=["Award Amount"])

# Feature 1: Log award amount
df["log_award_amount"] = np.log1p(df["Award Amount"])

# Feature 2: Is DoD
df["is_dod"] = df["Awarding Agency"].str.contains(
    "Defense", case=False, na=False
).astype(int)

# Feature 3: Contract duration
df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
df["End Date"] = pd.to_datetime(df["End Date"], errors="coerce")
df["contract_duration_days"] = (df["End Date"] - df["Start Date"]).dt.days

# Feature 4: Recipient experience
recipient_counts = df["Recipient Name"].value_counts()
df["recipient_experience"] = df["Recipient Name"].map(recipient_counts)

# Feature 5: Agency activity level
agency_counts = df["Awarding Agency"].value_counts()
df["agency_activity"] = df["Awarding Agency"].map(agency_counts)

# Feature 6: Is DC area performance
df["is_dc"] = df["Place of Performance State Code"].isin(
    ["DC", "VA", "MD"]
).astype(int)

# Feature 7: Award year
df["award_year"] = df["Start Date"].dt.year

# Feature 8: Award quarter — federal Q4 (July-Sept) is historically busiest
df["award_quarter"] = df["Start Date"].dt.quarter
df["is_q4"] = (df["award_quarter"] == 4).astype(int)

# Feature 9: Award amount bucket
df["amount_bucket_encoded"] = pd.cut(
    df["Award Amount"],
    bins=[0, 1000000, 5000000, 15000000, 50000001],
    labels=[0, 1, 2, 3]
).astype(float)

# Drop rows missing critical features
df = df.dropna(subset=[
    "contract_duration_days",
    "recipient_experience",
    "agency_activity",
    "award_year"
])

print("\nFinal shape:", df.shape)

feature_cols = [
    "log_award_amount", "is_dod", "contract_duration_days",
    "recipient_experience", "agency_activity",
    "is_dc", "award_year", "is_q4", "amount_bucket_encoded"
]

print("\nSample features:")
print(df[feature_cols + ["won_definitive"]].head())

df.to_csv("data/processed/contracts_features.csv", index=False)
print("\nSaved to data/processed/contracts_features.csv")