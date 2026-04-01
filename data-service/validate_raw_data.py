import pandas as pd

deliveries = pd.read_csv("../New Data/deliveries_updated_ipl_upto_2025.csv")
matches = pd.read_csv("../New Data/matches_updated_ipl_upto_2025.csv")

print("Deliveries shape:", deliveries.shape)
print("Matches shape:", matches.shape)

print("\nMissing values in deliveries")
print(deliveries.isnull().sum())

print("\nMissing values in matches")
print(matches.isnull().sum())

print("\nUnique matchId in deliveries:", deliveries["matchId"].nunique())
print("Unique matchId in matches:", matches["matchId"].nunique())

missing_matches = set(deliveries["matchId"]) - set(matches["matchId"])

print("\nMatchIds in deliveries not in matches:", len(missing_matches))

duplicates = deliveries.duplicated().sum()

print("\nDuplicate rows in deliveries:", duplicates)
