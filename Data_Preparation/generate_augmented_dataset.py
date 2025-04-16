import sys
import os

# Set base_dir based on script location, going up one level from the script's directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Ensure the base directory exists
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"Base directory not found: {base_dir}. Please ensure the project is located at this path.")

import pandas as pd
import importlib
import src.data_io
importlib.reload(src.data_io)
from src.data_io import load_questions, load_misspellings

import itertools

# Function to adjust the case of a word based on the original text
def adjust_case(original, new):
    if original.istitle():
        return new.capitalize()
    elif original.isupper():
        return new.upper()
    elif original.islower():
        return new.lower()
    return new

# Function to generate augmented variants of a sentence with misspellings
def generate_augmented_variants(sentence, misspellings_dict, max_errors=10):
    words = sentence.split()
    candidate_indices = [i for i, word in enumerate(words) if word.lower() in misspellings_dict]
    variants = [(sentence, 0)]
    max_errors = min(max_errors, len(candidate_indices))

    for error_count in range(1, max_errors + 1):
        for indices in itertools.combinations(candidate_indices, error_count):
            replacement_options = []
            for i in indices:
                word = words[i]
                candidates = misspellings_dict[word.lower()]
                adjusted_candidates = [adjust_case(word, cand) for cand in candidates]
                replacement_options.append(adjusted_candidates)

            for replacements in itertools.product(*replacement_options):
                new_words = words.copy()
                for idx, i in enumerate(indices):
                    new_words[i] = replacements[idx]
                variant = " ".join(new_words)
                variants.append((variant, error_count))

    return variants

# Function to check the version of the generate_augmented_variants function
def check_function_version():
    test_sentence = "Test"
    test_dict = {"test": ["tset", "tets"]}
    result = generate_augmented_variants(test_sentence, test_dict, max_errors=1)
    if len(result) > 1:
        print("Function version check: Updated version detected (multiple variants generated)")
    else:
        print("Function version check: Issue detected (expected multiple variants)")

check_function_version()

# Define data directory
data_dir = os.path.join(base_dir, 'data')

# Create data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Try to list directory contents and handle potential errors
try:
    files_in_data = os.listdir(data_dir)
except OSError as e:
    print(f"Error listing directory {data_dir}: {e}")
    raise

questions_path = os.path.join(base_dir, 'data', 'questions.csv')
misspellings_path = os.path.join(base_dir, 'data', 'misspellings.csv')
output_path = os.path.join(base_dir, 'data', 'augmented_questions.csv')

# Create default files if they don't exist
if not os.path.exists(questions_path):
    with open(questions_path, 'w') as f:
        f.write("question\nWhat is the capital of France?")
    print(f"Created default questions.csv at {questions_path}")

if not os.path.exists(misspellings_path):
    with open(misspellings_path, 'w') as f:
        f.write("word,misspellings\nwhat,\"wath,wtah\"\nis,iss")
    print(f"Created default misspellings.csv at {misspellings_path}")

# Load the questions and misspellings datasets
questions_df = load_questions(questions_path)
print(f"Loaded questions: {questions_df.shape}")

misspellings_dict = load_misspellings(misspellings_path)

questions_full = questions_df.copy()

# Generate augmented variants by combining the two datasets
augmented_variants = []
for idx, row in questions_full.iterrows():
    original_text = row['question']
    variants = generate_augmented_variants(original_text, misspellings_dict, max_errors=10)
    for variant_text, error_count in variants:
        augmented_variants.append({
            'original_question': original_text,
            'variant_question': variant_text,
            'error_count': error_count
        })

augmented_df = pd.DataFrame(augmented_variants)
print(f"Augmented dataset shape: {augmented_df.shape}")

# Save the augmented dataset to a new CSV file
try:
    augmented_df.to_csv(output_path, index=False)
    print(f"Augmented questions saved at {output_path}")
except Exception as e:
    print(f"Error saving augmented_questions.csv: {e}")
    raise