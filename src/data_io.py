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
    print(f"DEBUG - Entering load_misspellings with filepath: {filepath}")
    abs_filepath = os.path.join(project_root, filepath)
    print(f"DEBUG - Loading CSV from: {abs_filepath}")

    misspellings = {}
    with open(abs_filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        print(f"DEBUG - CSV header: {header}")
        for row in reader:
            print(f"DEBUG - Raw row: {row}")
            if len(row) != 2:
                print(f"DEBUG - Skipping invalid row (expected 2 fields, got {len(row)}): {row}")
                continue
            key = row[0].strip().lower()
            candidates = [s.strip() for s in row[1].split(",") if s.strip()]
            misspellings[key] = candidates
            print(f"DEBUG - Loaded: {key} -> {candidates}")

    print(f"Loaded misspellings dictionary with {len(misspellings)} entries.")
    return misspellings