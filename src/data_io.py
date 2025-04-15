import pandas as pd


def load_questions(filepath):
    """
    Load questions from a CSV file that contains a 'question' column.
    """
    df = pd.read_csv(filepath)
    if 'question' not in df.columns:
        raise ValueError("CSV file must contain a 'question' column.")
    return df


def load_misspellings(filepath):
    """
    Load misspellings from a CSV file.

    Expected CSV format:
      correct_word,misspellings
      What,"wath,wtah"

    Returns a dictionary mapping each correct word (in lowercase)
    to a list of candidate misspellings.
    """
    df = pd.read_csv(filepath, dtype=str)
    misspellings = {}
    for index, row in df.iterrows():
        key = row["correct_word"].strip().lower()
        # Split the comma-separated string into a list of candidates.
        cell_value = row["misspellings"]
        candidates = [s.strip() for s in cell_value.split(",") if s.strip()]
        misspellings[key] = candidates
    # Debug print to verify the dictionary contents.
    print("DEBUG - Loaded misspellings dictionary:", misspellings)
    return misspellings