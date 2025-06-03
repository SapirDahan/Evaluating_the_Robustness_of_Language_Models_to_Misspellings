import pandas as pd
import os
import csv

# Set paths
current_dir = os.path.dirname(__file__)
input_path = os.path.join(current_dir, 'misspelled.csv')
output_path = os.path.join(current_dir, '..', 'Data', 'misspellings.csv')

# Load the CSV
df = pd.read_csv(input_path)

# Rename columns if needed
if df.columns[0] == "":
    df.columns = ['index', 'correct_word', 'misspellings']
else:
    df = df.rename(columns={
        df.columns[0]: 'index',
        'label': 'correct_word',
        'input': 'misspellings'
    })

# Drop rows with missing misspellings
df = df.dropna(subset=['misspellings'])

# Group by correct word and aggregate misspellings
grouped = (
    df.groupby('correct_word')['misspellings']
    .apply(lambda x: ",".join(sorted(set(str(s).strip() for s in x))))
    .reset_index()
)

# Save with correct quoting (each field in one pair of quotes)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
grouped.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
print(f"Saved file to: {output_path}")