import pandas as pd
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_questions(filepath):
    abs_filepath = os.path.join(project_root, filepath)
    df = pd.read_csv(abs_filepath)
    if 'question' not in df.columns:
        raise ValueError("CSV file must contain a 'question' column.")
    return df

def load_misspellings(filepath):
    abs_filepath = os.path.join(project_root, filepath)
    df = pd.read_csv(abs_filepath, dtype=str, keep_default_na=False)
    misspellings = {}
    for _, row in df.iterrows():
        key = row["correct_word"].strip().lower()
        candidates = [s.strip() for s in row["misspellings"].split(",") if s.strip()]
        misspellings[key] = candidates
    print(f"Loaded misspellings dictionary with {len(misspellings)} entries.")
    return misspellings