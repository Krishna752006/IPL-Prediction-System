import numpy as np
import pandas as pd

df1 = pd.read_csv("deliveries_updated_ipl_upto_2025.csv")
df2 = pd.read_csv("matches_updated_ipl_upto_2025.csv")
print("DF1 Columns = ", df1.columns)
print("DF1 first 5 rows\n", df1.head())
print("\nDF2 Columns = ", df2.columns)
print("DF2 first 5 rows\n", df2.head())
