import pandas as pd
import os
import csv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_questions(filepath):
    abs_filepath = os.path.join(project_root, filepath)
    df = pd.read_csv(abs_filepath)
    if 'question' not in df.columns:
        raise ValueError("CSV file must contain a 'question' column.")
    return df


def load_misspellings(filepath):
    abs_filepath = os.path.join(project_root, filepath)

    misspellings = {}
    with open(abs_filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) != 2:
                continue
            key = row[0].strip().lower()
            candidates = [s.strip() for s in row[1].split(",") if s.strip()]
            misspellings[key] = candidates

    print(f"Loaded misspellings dictionary with {len(misspellings)} entries.")
    return misspellings