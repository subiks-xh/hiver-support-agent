import pandas as pd
import numpy as np

# Load dataset
print("Loading dataset... (may take 30-60 seconds)")
df = pd.read_csv('data/twcs/twcs.csv')

print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSample rows:")
print(df.head(5).to_string())

print(f"\nNull counts:")
print(df.isnull().sum())

print(f"\ninbound value counts:")
print(df['inbound'].value_counts())

# Find all brand accounts (inbound=False means brand is responding)
print("\n--- Finding all brands ---")
support_responses = df[df['inbound'] == False]
brand_counts = support_responses['author_id'].value_counts()

print(f"\nTop 30 brands by response count:")
print(brand_counts.head(30))

# Save
brand_counts.to_csv('data/brand_counts.csv')
print("\nSaved brand_counts.csv")